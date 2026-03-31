#!/usr/bin/env python3
"""
Causal LM pretraining from scratch (~400M params, Llama-style) with streaming HF datasets.

Defaults target a ~12h-friendly run on 1x 80GB GPU (2k packed length, 10B tokens). Use ``--fast``
on H100 (Hopper) for batching/prefetch/attention/compile settings aimed at maximum throughput.

Architecture mirrors *ratios* common in modern dense LMs (e.g. Qwen3-style GQA, ~3× MLP width,
large RoPE base, RMSNorm eps 1e-6, tied embeddings). A 400M model cannot literally match a
1.3B-class model on every benchmark; the goal is strong quality per parameter plus follow-up SFT.

DeepSeek-style inspirations applied here (without MoE/MLA/FP8 infra):
- **Data**: optional stream interleaving toward higher-quality / cleaner text (default: small Wikitext mix).
- **Denser supervision**: optional mid-layer CE via final norm+lm_head (lighter cousin of multi-token / deep supervision).
- **Optimizer**: no weight decay on norm/bias parameters (common in modern LM training).
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import threading
import time
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import interleave_datasets, load_dataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    SchedulerType,
    get_scheduler,
)


def _configure_cuda_performance() -> None:
    """Bias PyTorch toward fast matmul/attention paths on NVIDIA GPUs (A100/H100)."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    # Prefer flash / memory-efficient SDPA kernels when shapes allow (PyTorch 2.x).
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except AttributeError:
        pass


def _try_upgrade_sdpa_to_flash(current: str) -> str:
    """If using PyTorch SDPA path, prefer FlashAttention-2 in Transformers when ``flash_attn`` is installed."""
    if current != "sdpa":
        return current
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def apply_fast_gpu_settings(args: argparse.Namespace) -> None:
    """
    Hopper-oriented throughput presets. Implies BF16 on CUDA, larger microbatch/prefetch,
    optional FlashAttention-2, and torch.compile when compatible with other flags.
    """
    if not getattr(args, "fast", False):
        return
    if torch.cuda.is_available():
        args.bf16 = True
    args.prefetch_batches = max(args.prefetch_batches, 16)
    args.tokenize_batch_size = max(args.tokenize_batch_size, 128)
    # H100 80GB + ~400M @ 2k: 16 sequences microbatch is usually safe with gradient checkpointing.
    if args.per_device_train_batch_size <= 8:
        args.per_device_train_batch_size = 16
    args.attn_implementation = _try_upgrade_sdpa_to_flash(args.attn_implementation)
    if args.dense_supervision_lambda > 0:
        args.compile = False
    elif not getattr(args, "no_compile", False):
        args.compile = True
    if args.dense_supervision_lambda > 0:
        warnings.warn(
            "--fast + dense_supervision_lambda>0 disables torch.compile. "
            "For maximum H100 throughput use --dense_supervision_lambda 0.",
            UserWarning,
            stacklevel=2,
        )


def _iterable_prefetch(it: Iterator[Dict[str, torch.Tensor]], prefetch_batches: int) -> Iterator[Dict[str, torch.Tensor]]:
    """Prefetch DataLoader batches on a background thread so tokenization/IO can overlap GPU work."""
    if prefetch_batches <= 1:
        yield from it
        return

    q: queue.Queue = queue.Queue(maxsize=prefetch_batches)
    sentinel: object = object()

    def worker() -> None:
        try:
            for batch in it:
                q.put(batch)
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item


class StreamingLMIterable(IterableDataset):
    """Tokenize a streaming HF dataset and yield fixed-length blocks (infinite restart)."""

    def __init__(
        self,
        stream_factory: Callable[[], Any],
        tokenizer,
        block_size: int,
        text_column: str,
        seed: int,
        shuffle_buffer_size: int,
        tokenize_batch_size: int,
    ) -> None:
        super().__init__()
        self.stream_factory = stream_factory
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_column = text_column
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.tokenize_batch_size = max(1, int(tokenize_batch_size))
        self.eos_id: Optional[int] = tokenizer.eos_token_id

    def _flush_text_batch(self, texts: List[str], buffer: List[int]) -> None:
        if not texts:
            return
        enc = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        for ids in enc["input_ids"]:
            if not ids:
                continue
            buffer.extend(ids)
            if self.eos_id is not None:
                buffer.append(self.eos_id)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 0:
            raise RuntimeError("Use num_workers=0 for this IterableDataset (streaming).")

        epoch = 0
        while True:
            raw = self.stream_factory()
            ds = raw.shuffle(seed=self.seed + epoch, buffer_size=self.shuffle_buffer_size)
            it = iter(ds)
            buffer: List[int] = []
            buf_start = 0
            text_batch: List[str] = []

            def _compact() -> None:
                nonlocal buffer, buf_start
                if buf_start <= 0:
                    return
                if buf_start >= len(buffer):
                    buffer.clear()
                    buf_start = 0
                    return
                if buf_start > 65536 or buf_start > len(buffer) // 2:
                    buffer = buffer[buf_start:]
                    buf_start = 0

            def drain_buffer() -> Iterator[Dict[str, torch.Tensor]]:
                nonlocal buf_start
                while len(buffer) - buf_start >= self.block_size:
                    s = buf_start
                    e = s + self.block_size
                    chunk = buffer[s:e]
                    buf_start = e
                    _compact()
                    block = torch.as_tensor(chunk, dtype=torch.long)
                    yield {"input_ids": block, "labels": block}

            while True:
                try:
                    ex = next(it)
                except StopIteration:
                    break

                text = ex.get(self.text_column)
                if not text or not isinstance(text, str):
                    continue

                text_batch.append(text)
                if len(text_batch) >= self.tokenize_batch_size:
                    self._flush_text_batch(text_batch, buffer)
                    text_batch.clear()
                    yield from drain_buffer()

            self._flush_text_batch(text_batch, buffer)
            text_batch.clear()
            yield from drain_buffer()

            epoch += 1


def llama_config_roughly_400m(vocab_size: int, max_position_embeddings: int) -> LlamaConfig:
    """
    ~400M-parameter dense decoder (total count depends on vocab via tied embeddings).

    Shaped after public Qwen3 dense patterns (GQA 2:1, MLP ≈ 3× hidden, rope_theta 1e6, rms 1e-6,
    tied embeddings) while using a standard LlamaForCausalLM implementation in Transformers.
    """
    hidden_size = 1024
    num_attention_heads = 16
    num_key_value_heads = 8
    intermediate_size = 3 * hidden_size  # 3072; same width ratio as Qwen3-1.7B (6144/2048)

    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=24,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_bias=False,
        tie_word_embeddings=True,
        use_cache=False,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream-pretrain a small Llama-style LM.")

    p.add_argument("--output_dir", type=str, default="./checkpoints/pretrain")
    p.add_argument("--tokenizer_name", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

    p.add_argument("--dataset_name", type=str, default="allenai/c4")
    p.add_argument("--dataset_config", type=str, default="en")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument(
        "--mix_probability",
        type=float,
        default=0.08,
        help="Sample this fraction from --mix_dataset_* (higher-quality / cleaner mix). 0 disables.",
    )
    p.add_argument("--mix_dataset_name", type=str, default="wikitext")
    p.add_argument("--mix_dataset_config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--mix_dataset_split", type=str, default="train")
    p.add_argument("--mix_text_column", type=str, default="text")

    p.add_argument(
        "--dense_supervision_lambda",
        type=float,
        default=0.1,
        help="Mid-layer auxiliary CE weight (denser training signal; set 0 to disable). May conflict with --compile.",
    )

    p.add_argument("--block_size", type=int, default=2048, help="Packed sequence length (context window for RoPE).")
    p.add_argument("--shuffle_buffer_size", type=int, default=10_000)
    p.add_argument(
        "--tokenize_batch_size",
        type=int,
        default=64,
        help="Number of text rows encoded per tokenizer call (Rust fast path).",
    )
    p.add_argument(
        "--prefetch_batches",
        type=int,
        default=8,
        help="CPU batches to queue ahead of the GPU (1 disables prefetch thread).",
    )

    p.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Microbatch sequences; 2k default allows a larger microbatch than 4k on an A100 80GB.",
    )
    p.add_argument("--gradient_accumulation_steps", type=int, default=64)

    p.add_argument("--total_tokens_b", type=float, default=10.0, help="Total training tokens in billions.")
    p.add_argument("--max_steps", type=int, default=None, help="Override step budget (if set, ignores total_tokens_b).")

    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=[e.value for e in SchedulerType],
    )
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Also print a full text line to stdout every N optimizer steps (tqdm bar updates every step).",
    )
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    p.add_argument("--bf16", action="store_true", help="Use bf16 (recommended on A100/H100).")
    p.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--disable_gradient_checkpointing", action="store_true")
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model (PyTorch 2+; can interact badly with some checkpointing setups).",
    )
    p.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile even when using --fast.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="H100/Hopper-oriented: BF16 on CUDA, larger batch/prefetch/tokenize, FlashAttn2 if installed, "
        "torch.compile when dense_supervision is off. OOM → lower --per_device_train_batch_size.",
    )

    return p.parse_args()


def validate_mix_stream(args: argparse.Namespace, tokenizer: Any) -> None:
    if args.mix_probability <= 0:
        return
    mcfg = args.mix_dataset_config or None
    if mcfg == "":
        mcfg = None
    ds = load_dataset(
        args.mix_dataset_name,
        mcfg,
        split=args.mix_dataset_split,
        streaming=True,
        trust_remote_code=args.trust_remote_code,
    )
    it = iter(ds)
    for _ in range(512):
        try:
            ex = next(it)
        except StopIteration:
            raise RuntimeError("Mix streaming dataset produced no examples.") from None
        col = args.mix_text_column or args.text_column
        if col not in ex:
            keys = list(ex.keys()) if isinstance(ex, dict) else []
            raise ValueError(f"Mix --mix_text_column {col!r} not in keys {keys}.")
        text = ex.get(col)
        if isinstance(text, str) and text.strip():
            if tokenizer.encode(text, add_special_tokens=False):
                return
    raise ValueError(f"No usable text in mix dataset (column {col!r}).")


def validate_streaming_dataset(args: argparse.Namespace, tokenizer: Any) -> None:
    """Fail fast if the text column is wrong or rows never yield strings (avoids a stuck dataloader)."""
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
        streaming=True,
        trust_remote_code=args.trust_remote_code,
    )
    it = iter(ds)
    for _ in range(512):
        try:
            ex = next(it)
        except StopIteration:
            raise RuntimeError("Streaming dataset produced no examples.") from None
        if args.text_column not in ex:
            keys = list(ex.keys()) if isinstance(ex, dict) else []
            raise ValueError(
                f"--text_column {args.text_column!r} not in example keys {keys}. "
                "Wrong column name skips every row and training will hang."
            )
        text = ex.get(args.text_column)
        if isinstance(text, str) and text.strip():
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            return
    raise ValueError(
        f"No usable text in first 512 examples (column {args.text_column!r}). "
        "Check dataset split, filters, and column name."
    )


def _build_adamw(model: Any, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW with no decay on biases / norms (standard for Transformer LMs). Fused on CUDA when available."""
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if param.ndim == 1 or "bias" in lname or "norm" in lname:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = dict(params=groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(**kwargs, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(**kwargs)


def main() -> None:
    # Helps long-run allocator behavior; safe before first CUDA alloc.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parse_args()
    apply_fast_gpu_settings(args)
    set_seed(args.seed)
    _configure_cuda_performance()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if args.bf16 else None,
    )

    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        validate_streaming_dataset(args, tokenizer)
        validate_mix_stream(args, tokenizer)

    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()

    config = llama_config_roughly_400m(
        vocab_size=len(tokenizer),
        max_position_embeddings=args.block_size,
    )

    attn_kwargs: Dict[str, Any] = {"attn_implementation": args.attn_implementation}

    if args.resume_from_checkpoint:
        model = LlamaForCausalLM.from_pretrained(
            args.resume_from_checkpoint,
            torch_dtype=torch.bfloat16 if args.bf16 else None,
            **attn_kwargs,
        )
    else:
        model = LlamaForCausalLM(config, **attn_kwargs)

    if not args.disable_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    do_compile = bool(args.compile) and not getattr(args, "no_compile", False)
    if do_compile and args.dense_supervision_lambda > 0:
        if accelerator.is_main_process:
            accelerator.print(
                "torch.compile disabled while dense_supervision_lambda > 0 (mid-layer forward hooks)."
            )
        do_compile = False

    if do_compile:
        try:
            model = torch.compile(model, dynamic=False, mode="max-autotune")
        except TypeError:
            model = torch.compile(model, dynamic=False)

    if accelerator.is_main_process and getattr(args, "fast", False):
        accelerator.print(
            f"H100/FAST: bf16={args.bf16} microbatch={args.per_device_train_batch_size} "
            f"prefetch={args.prefetch_batches} tokenizer_batch={args.tokenize_batch_size} "
            f"attn={args.attn_implementation} torch.compile={do_compile} | "
            f"install flash-attn for best Hopper attention throughput"
        )

    if accelerator.is_local_main_process:
        nparams = sum(p.numel() for p in model.parameters())
        accelerator.print(f"Model parameters: {nparams / 1e6:.2f}M")

    tokens_per_step = (
        args.per_device_train_batch_size
        * args.block_size
        * args.gradient_accumulation_steps
        * accelerator.num_processes
    )

    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        total_tokens = int(args.total_tokens_b * 1e9)
        max_steps = math.ceil(total_tokens / tokens_per_step)

    use_mix = args.mix_probability > 0
    accelerator.print(
        f"Tokens/step (global): {tokens_per_step:,} | "
        f"Steps for {args.total_tokens_b}B tokens: {max_steps:,}"
    )
    if accelerator.is_main_process:
        if use_mix:
            accelerator.print(
                f"Stream mix: {(1 - args.mix_probability) * 100:.1f}% {args.dataset_name} / "
                f"{args.mix_probability * 100:.1f}% {args.mix_dataset_name}"
            )
        if args.dense_supervision_lambda > 0:
            accelerator.print(f"Dense supervision λ={args.dense_supervision_lambda} (mid-layer aux CE)")

    text_column_effective = "text" if use_mix else args.text_column

    def stream_factory() -> Any:
        main_ds = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split,
            streaming=True,
            trust_remote_code=args.trust_remote_code,
        )
        if use_mix:
            mcfg = args.mix_dataset_config or None
            if mcfg == "":
                mcfg = None
            mix_ds = load_dataset(
                args.mix_dataset_name,
                mcfg,
                split=args.mix_dataset_split,
                streaming=True,
                trust_remote_code=args.trust_remote_code,
            )
            tcol = args.text_column
            mcol = args.mix_text_column or args.text_column

            def norm_main(ex: Dict[str, Any]) -> Dict[str, str]:
                v = ex.get(tcol)
                return {"text": v if isinstance(v, str) else ""}

            def norm_mix(ex: Dict[str, Any]) -> Dict[str, str]:
                v = ex.get(mcol)
                return {"text": v if isinstance(v, str) else ""}

            try:
                main_ds = main_ds.map(norm_main, remove_columns=list(main_ds.column_names))
            except Exception:
                main_ds = main_ds.map(norm_main)
            try:
                mix_ds = mix_ds.map(norm_mix, remove_columns=list(mix_ds.column_names))
            except Exception:
                mix_ds = mix_ds.map(norm_mix)

            ds = interleave_datasets(
                [main_ds, mix_ds],
                probabilities=[1.0 - args.mix_probability, args.mix_probability],
                seed=args.seed,
                stopping_strategy="first_exhausted",
            )
        else:
            ds = main_ds

        if accelerator.num_processes > 1:
            ds = ds.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)
        return ds

    train_ds = StreamingLMIterable(
        stream_factory=stream_factory,
        tokenizer=tokenizer,
        block_size=args.block_size,
        text_column=text_column_effective,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        tokenize_batch_size=args.tokenize_batch_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = _build_adamw(model, args.learning_rate, args.weight_decay)

    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * max_steps),
        num_training_steps=max_steps,
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    model.train()
    global_step = 0
    micro_loss_sum = 0.0
    ema_step_s: Optional[float] = None

    progress = tqdm(
        total=max_steps,
        desc="train",
        unit="step",
        dynamic_ncols=True,
        miniters=1,
        mininterval=0.0,
        smoothing=0.05,
        disable=not accelerator.is_main_process,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    base_iter = iter(train_loader)
    data_iter = _iterable_prefetch(base_iter, args.prefetch_batches)

    step_started = time.perf_counter()

    while global_step < max_steps:
        with accelerator.accumulate(model):
            batch = next(data_iter)
            inner = accelerator.unwrap_model(model)

            if args.dense_supervision_lambda > 0:
                mid = len(inner.model.layers) // 2
                aux_hidden: List[Optional[torch.Tensor]] = [None]

                def _hook(_mod: Any, _inp: Any, out: Any) -> None:
                    aux_hidden[0] = out[0]

                hnd = inner.model.layers[mid].register_forward_hook(_hook)
                try:
                    out = model(**batch)
                finally:
                    hnd.remove()

                loss_main = out.loss
                if aux_hidden[0] is not None:
                    logits_aux = inner.lm_head(inner.model.norm(aux_hidden[0]))
                    logits_f = logits_aux[:, :-1, :].contiguous()
                    labels_s = batch["labels"][:, 1:].contiguous()
                    loss_aux = F.cross_entropy(
                        logits_f.float().reshape(-1, logits_f.size(-1)),
                        labels_s.reshape(-1),
                    )
                    loss = loss_main + args.dense_supervision_lambda * loss_aux
                else:
                    loss = loss_main
            else:
                out = model(**batch)
                loss = out.loss

            accelerator.backward(loss)
            micro_loss_sum += loss.detach().float().item()

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            avg_micro_loss = micro_loss_sum / max(args.gradient_accumulation_steps, 1)
            micro_loss_sum = 0.0

            global_step += 1
            step_elapsed = max(time.perf_counter() - step_started, 1e-9)
            step_started = time.perf_counter()

            if ema_step_s is None:
                ema_step_s = step_elapsed
            else:
                ema_step_s = 0.08 * step_elapsed + 0.92 * ema_step_s

            tok_s = tokens_per_step / ema_step_s
            lr = scheduler.get_last_lr()[0]

            if accelerator.is_main_process:
                progress.set_postfix(
                    loss=f"{avg_micro_loss:.4f}",
                    lr=f"{lr:.2e}",
                    tokM=f"{tok_s/1e6:.3f}",
                    rem=f"{max_steps - global_step}",
                    refresh=False,
                )
                progress.update(1)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    remaining_steps = max_steps - global_step
                    eta_s = ema_step_s * remaining_steps
                    eta_h, rem = divmod(int(eta_s), 3600)
                    eta_m, eta_sec = divmod(rem, 60)
                    tokens_done = global_step * tokens_per_step
                    print(
                        f"[step {global_step}/{max_steps}] "
                        f"loss={avg_micro_loss:.4f} lr={lr:.2e} "
                        f"throughput={tok_s/1e6:.3f}M tok/s "
                        f"tokens_seen≈{tokens_done/1e9:.3f}B "
                        f"eta≈{eta_h:d}h{eta_m:02d}m{eta_sec:02d}s",
                        flush=True,
                    )

            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"step_{global_step}")
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(save_path, safe_serialization=True)
                    tokenizer.save_pretrained(save_path)
                    accelerator.print(f"Saved checkpoint to {save_path}")

    progress.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "final")
        # Wrapped torch.compile / DDP: save unwrapped state_dict is handled inside save_pretrained
        accelerator.unwrap_model(model).save_pretrained(final_path, safe_serialization=True)
        tokenizer.save_pretrained(final_path)
        accelerator.print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
