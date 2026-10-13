#!/usr/bin/env python3
"""
====================================================================
 Weight-Compression Benchmark — M0 / M1 / M2 / M3(BoseComp) / BOSESIR
 (v8 — bugfix revision: gate-dtype grad fix + left-truncation fix)
====================================================================

This is FILE 1 of a two-file split. FILE 2 (kvcache_bench.py) covers
M4/WormComp + other KV-cache methods and is intentionally separate: this
file only ever touches WEIGHT storage, never the KV cache.

--------------------------------------------------------------------
WHAT CHANGED IN THIS REVISION (v8), ON TOP OF v7 — READ THIS
--------------------------------------------------------------------
This revision fixes two bugs identified in external code review of v7,
plus the removal of the fabricated MuseGlimmer-30B, model registry, and
no-silent-fallback behavior already introduced in v7 (unchanged here).

BUG FIX 1 — BOSESIR gate-hook gradient-disconnection bug
  In v7, `_LayerGateHook.make_gate` created the gate tensor as
  `torch.float32` unconditionally:
      self.gate = torch.ones((), device=device, dtype=torch.float32, ...)
  and then in the forward hook it was cast to the activation's dtype
  before multiplying:
      g = self.gate.to(dtype=output.dtype)
      gated = output * g
  When the model runs in bfloat16 (the default `dtype` for every model in
  MODEL_REGISTRY), `self.gate.to(dtype=torch.bfloat16)` creates a new,
  non-leaf tensor `g` in the autograd graph. Gradients flowing backward
  accumulate on `g`, not on the leaf `self.gate` — so `self.gate.grad`
  frequently evaluates to `None` after `loss.backward()`, silently
  zeroing the per-layer utility signal `U[l]` for every layer, every
  step. This would have made BOSESIR's evidence loop return a near-zero
  or degenerate utility signal without raising any error.

  FIX: the gate tensor is now created directly in the model's actual
  compute dtype (passed in from `bose_evidence_loop`, sourced from
  `next(model.parameters()).dtype`) so it IS the leaf that participates
  directly in the multiply — no intermediate cast node, no disconnection.
  A runtime assertion (`_assert_gate_grads_nonzero`) now runs once per
  evidence loop and raises if every captured `u_raw` value came out as
  exactly 0.0 across an entire step, so a similar silent-degeneration bug
  in the future fails loudly instead of quietly returning garbage.

BUG FIX 2 — GSM8K few-shot prompt truncation could sever the question
  In v7, `tokenizer(..., truncation=True, max_length=2048)` used the
  tokenizer's default truncation side (right-side truncation for most
  causal LM tokenizers, including Llama-3 and Qwen-2.5's fast
  tokenizers). With a 5-shot prefix prepended, a long few-shot block
  could push the actual target question past the 2048-token budget,
  right-truncating the question itself (or its "Answer:" cue) off the
  end of the prompt — silently corrupting the eval example rather than
  trimming the (less important) earlier few-shot demonstrations.

  FIX: `tokenizer.truncation_side = "left"` is now explicitly set before
  the GSM8K few-shot generation loop (and restored afterward), so
  truncation trims from the *front* of the prompt (the earliest few-shot
  examples) and the actual target question + "Answer:" cue at the end is
  always preserved intact.

Everything else (model registry, real-data-only calibration, ICD-10
synthetic-probe relabeling, PackedLinear bit-packing, Fermi-Dirac M3
cooling sweep, Bose-Einstein BOSESIR bisection) is unchanged from v7 and
is NOT re-explained here — see inline comments below for that logic.

--------------------------------------------------------------------
SETUP
--------------------------------------------------------------------
pip install torch transformers accelerate datasets evaluate rouge_score \
            numpy nvidia-ml-py sentencepiece protobuf --break-system-packages

Hugging Face auth is REQUIRED for Llama-3.1 and Mistral-7B (gated repos):
    export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
If HF_TOKEN is unset and a gated repo is requested, this script raises
ModelLoadError immediately rather than attempting a doomed anonymous
download.

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
Single arm:
    python weight_compression_bench.py --model llama-3.1-8b --condition M3
    python weight_compression_bench.py --model kimi-linear-48b --condition BOSESIR

Full sweep, resumable (valid results3/{model}__{condition}.json files are
skipped unless --force / --force-conditions is given):
    python weight_compression_bench.py --sweep
    python weight_compression_bench.py --sweep --models llama-3.1-8b mistral-7b
    python weight_compression_bench.py --sweep --force
    python weight_compression_bench.py --sweep --force-conditions M3 BOSESIR

Aggregate existing results3/*.json into the reporting-matrix CSVs:
    python weight_compression_bench.py --aggregate-only

All paths (results3/, hf_cache/, logs/) are RELATIVE to the current
working directory.

--------------------------------------------------------------------
DATASETS (all real, public, on Hugging Face)
--------------------------------------------------------------------
  SQuAD v2 (extractive QA / unanswerable detection): rajpurkar/squad_v2
  SST-2 (sentiment classification):                  stanfordnlp/sst2
  GSM8K (grade-school math, chain-of-thought):       openai/gsm8k
  CNN/DailyMail (summarization):                     abisee/cnn_dailymail
  ICD-10 SYNTHETIC PROBE (see module docstring):      atta00/icd10-codes
====================================================================
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("weight_compression_bench")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TORCH_AVAILABLE = False
    logger.warning("torch not usable in this environment — only --aggregate-only will work.")

if _TORCH_AVAILABLE and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TRANSFORMERS_AVAILABLE = False

try:
    import pynvml
    _NVML_AVAILABLE = True
except Exception:  # noqa: BLE001
    _NVML_AVAILABLE = False

try:
    from datasets import load_dataset
    _DATASETS_AVAILABLE = True
except Exception:  # noqa: BLE001
    _DATASETS_AVAILABLE = False

try:
    import evaluate as hf_evaluate
    _EVALUATE_AVAILABLE = True
except Exception:  # noqa: BLE001
    _EVALUATE_AVAILABLE = False


# ====================================================================
# EXPLICIT FAILURE MODES (no silent fallback, per user requirement)
# ====================================================================

class CalibrationDataUnavailableError(RuntimeError):
    """Raised instead of silently substituting fabricated calibration text."""


class ModelLoadError(RuntimeError):
    """Raised instead of silently substituting a fabricated model wrapper."""


class ProjectionCoverageError(RuntimeError):
    """Raised when a model's real architecture doesn't match this script's
    projection-name assumptions well enough to make bit-allocation results
    meaningful (see KIMI_LINEAR_CAVEATS in the module docstring)."""


class EvidenceSignalDegenerateError(RuntimeError):
    """Raised if BOSESIR's evidence loop produces an all-zero utility
    signal for an entire step, which previously happened silently due to
    the gate-dtype gradient-disconnection bug (see v8 changelog above).
    A real evidence signal should essentially never be exactly zero for
    every layer across an entire calibration batch."""


def _require_datasets_available():
    if not _DATASETS_AVAILABLE:
        raise CalibrationDataUnavailableError(
            "The `datasets` package is not importable. Install it "
            "(`pip install datasets --break-system-packages`) — this script "
            "does not substitute fabricated calibration text when real "
            "datasets are unavailable."
        )


# ====================================================================
# 1. CONFIG
# ====================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")  # no placeholder — None means "not set"

RESULTS_DIR = os.path.join("results3")
CACHE_DIR = os.path.join("hf_cache")
LOG_DIR = os.path.join("logs")
for _d in (RESULTS_DIR, CACHE_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ---- Model registry — every entry here is a real, existing HF repo as of
# Sept 2026 (verified by web search at authoring time; repos can still be
# renamed/removed/regated after the fact, in which case this script raises
# ModelLoadError with the underlying HF error rather than fabricating a
# model). `gated` and `trust_remote_code` are recorded explicitly per model
# rather than inferred, since guessing wrong either fails a public load
# with an unhelpful auth error or silently trusts remote code you didn't
# ask for.
@dataclass
class ModelSpec:
    repo_id: str
    gated: bool
    trust_remote_code: bool
    notes: str = ""


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec("meta-llama/Llama-3.1-8B-Instruct", gated=True, trust_remote_code=False),
    "qwen-2.5-7b": ModelSpec("Qwen/Qwen2.5-7B-Instruct", gated=False, trust_remote_code=False),
    "mistral-7b": ModelSpec("mistralai/Mistral-7B-Instruct-v0.3", gated=True, trust_remote_code=False),
    "kimi-linear-48b": ModelSpec(
        "moonshotai/Kimi-Linear-48B-A3B-Instruct", gated=False, trust_remote_code=True,
        notes="Custom hybrid linear-attention architecture (kimi_linear). "
              "See KIMI_LINEAR_CAVEATS in module docstring — projection-name "
              "coverage is checked and enforced at load time, M1/M2/M3/BOSESIR "
              "will refuse to run on it if coverage is too low to be meaningful."
    ),
}

CONDITIONS = ["M0", "M1", "M2", "M3", "BOSESIR"]
PROJECTION_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

TASK_SPECIFIC_CONDITIONS = {"M3": "m3", "BOSESIR": "bosesir"}

BIT_LATTICE = (8, 6, 5, 4, 3, 2, 0)

N_SST2 = 800
N_SQUAD = 500
N_GSM8K = 400
N_GSM8K_SHOT = 5
N_SUMM = 300
N_ICD10_SYNTHETIC_PROBE = 500
N_TASK_CALIB_PER_TASK = 64

# Minimum acceptable projection-name match coverage before M1/M2/M3/BOSESIR
# are allowed to run on a model (see ProjectionCoverageError / module
# docstring KIMI_LINEAR_CAVEATS).
MIN_PROJECTION_TYPES_MATCHED = 4       # out of len(PROJECTION_NAMES) == 7
MIN_MATCHED_LINEAR_FRACTION = 0.30     # matched Linear count / total Linear count


@dataclass
class BoseCompConfig:
    alpha: float = 0.4
    beta: float = 0.6
    k_max_frac: float = 0.15
    b_max: int = 8
    b_core: int = 16
    min_attn_bits: int = 2
    mu_start_frac: float = 1.4
    mu_end_frac: float = -0.1
    t0_frac: float = 0.12
    cooling_rate: float = 0.92
    loss_tolerance: float = 0.3
    max_cooling_steps: int = 60
    bit_lattice: tuple = BIT_LATTICE
    quant_block_size: int = 128


@dataclass
class BoseSirConfig:
    rho_u: float = 0.9
    rho_r: float = 0.9
    rho_i: float = 0.9
    rho_n: float = 0.9
    rho_d: float = 0.9
    alpha: float = 1.0
    beta: float = 1.0
    chi: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    energy_ablation: str = "full"
    n_evidence_steps: int = 12
    evidence_batch_size: int = 1
    max_len: int = 512
    T_B: float = 1.0
    n_f_frac: float = 0.5
    pole_eps: float = 1e-8
    mu_bisect_iters: int = 60
    mu_search_margin: float = 50.0
    core_layer_frac: float = 0.1
    quant_block_size: int = 128
    eps0: float = 1e-8
    min_attn_bits: int = 4
    novelty_clip_max: float = 10.0


BOSESIR_C_L = len(PROJECTION_NAMES)  # 7
QUANTA_TO_BITS = {0: 2, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 8}


@dataclass
class RunConfig:
    model_key: str
    condition: str
    calib_seq_len: int = 1024
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 0
    results_dir: str = RESULTS_DIR
    bosecomp: BoseCompConfig = field(default_factory=BoseCompConfig)
    bosesir: BoseSirConfig = field(default_factory=BoseSirConfig)
    task_n_calib_per_task: int = N_TASK_CALIB_PER_TASK
    bosesir_n_calib_per_task: Optional[int] = None

    def result_path(self) -> str:
        return os.path.join(self.results_dir, f"{self.model_key}__{self.condition}.json")

    def audit_path(self) -> str:
        return os.path.join(self.results_dir, f"{self.model_key}__{self.condition}__audit.json")

    def bosesir_n_calib(self) -> int:
        return self.bosesir_n_calib_per_task if self.bosesir_n_calib_per_task is not None else self.task_n_calib_per_task


def resolve_model_spec(model_key: str) -> ModelSpec:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key '{model_key}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_key]


# ====================================================================
# 2. MODEL LOAD / DELOAD — real models only, loud failure on gating/network
# ====================================================================

def _dtype_from_str(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_model_and_tokenizer(spec: ModelSpec, dtype: str = "bfloat16", device: str = "cuda",
                              cache_dir: str = CACHE_DIR):
    if spec.gated and not HF_TOKEN:
        raise ModelLoadError(
            f"'{spec.repo_id}' is a gated repo and HF_TOKEN is not set in the "
            f"environment. Request access on huggingface.co, then "
            f"`export HF_TOKEN=hf_xxx...` before running. This script does not "
            f"fall back to a fabricated model when a real load is blocked."
        )
    torch_dtype = _dtype_from_str(dtype)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.repo_id, token=HF_TOKEN, cache_dir=cache_dir,
            trust_remote_code=spec.trust_remote_code,
        )
    except Exception as e:
        raise ModelLoadError(f"Failed to load tokenizer for '{spec.repo_id}': {e}") from e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            spec.repo_id,
            token=HF_TOKEN,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map={"": device} if device == "cuda" else None,
            trust_remote_code=spec.trust_remote_code,
        )
    except Exception as e:
        raise ModelLoadError(
            f"Failed to load model '{spec.repo_id}' via AutoModelForCausalLM: {e}. "
            f"This script does not fabricate a substitute model when a real load "
            f"fails — fix the underlying issue (access request, trust_remote_code "
            f"requirement, disk space, network) and retry."
        ) from e

    model.eval()
    logger.info("Loaded %s onto %s (%s)", spec.repo_id, device, dtype)
    return model, tokenizer


def deload_model(model) -> None:
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("Model deloaded, CUDA cache cleared.")


@contextmanager
def managed_model(spec: ModelSpec, dtype: str = "bfloat16", device: str = "cuda", cache_dir: str = CACHE_DIR):
    model, tokenizer = load_model_and_tokenizer(spec, dtype=dtype, device=device, cache_dir=cache_dir)
    try:
        yield model, tokenizer
    finally:
        deload_model(model)


def peak_vram_gb(reset: bool = False) -> float:
    if not (_TORCH_AVAILABLE and torch.cuda.is_available()):
        return 0.0
    val = torch.cuda.max_memory_allocated() / 1e9
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return val


def current_vram_gb() -> float:
    if not (_TORCH_AVAILABLE and torch.cuda.is_available()):
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


# ====================================================================
# 2b. PROJECTION-NAME COVERAGE CHECK (guards against Kimi-Linear-style
#     architecture mismatch producing meaningless "compression" numbers)
# ====================================================================

def sanity_check_projection_match(model, model_key: str) -> dict:
    """
    Counts how many of PROJECTION_NAMES actually match a Linear submodule
    somewhere in the model, and what fraction of ALL Linear submodules in
    the model that match set covers. Raises ProjectionCoverageError if
    coverage is too low for M1/M2/M3/BOSESIR results to mean what their
    field names claim.
    """
    total_linear = 0
    matched_linear = 0
    matched_types = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_linear += 1
            for p in PROJECTION_NAMES:
                if name.endswith(p):
                    matched_linear += 1
                    matched_types.add(p)
                    break
    frac = matched_linear / max(total_linear, 1)
    report = {
        "model_key": model_key,
        "total_linear_modules": total_linear,
        "matched_linear_modules": matched_linear,
        "matched_fraction": frac,
        "matched_projection_types": sorted(matched_types),
        "n_matched_types": len(matched_types),
    }
    logger.info("Projection coverage check for %s: %s", model_key, report)
    if len(matched_types) < MIN_PROJECTION_TYPES_MATCHED or frac < MIN_MATCHED_LINEAR_FRACTION:
        raise ProjectionCoverageError(
            f"Model '{model_key}' matched only {len(matched_types)}/{len(PROJECTION_NAMES)} "
            f"projection name types ({sorted(matched_types)}) covering "
            f"{frac:.1%} of its {total_linear} Linear submodules "
            f"(need >= {MIN_PROJECTION_TYPES_MATCHED} types and >= "
            f"{MIN_MATCHED_LINEAR_FRACTION:.0%} coverage). This model's real "
            f"architecture does not match this script's q/k/v/o/gate/up/down "
            f"projection-naming assumption closely enough for M1/M2/M3/BOSESIR "
            f"bit-allocation and compression-ratio numbers to be meaningful — "
            f"see KIMI_LINEAR_CAVEATS in the module docstring. Refusing to run "
            f"this condition rather than silently quantizing an unrepresentative "
            f"subset of the model."
        )
    return report


# ====================================================================
# 3. REAL BIT-PACKED QUANTIZED STORAGE
# (verified correct via independent numpy re-implementation + round-trip
#  test across bits in {1..8} and n in {1,7,128,129,1000} — see
#  test_bitpack.py. Logic below is unchanged from that verified version.)
# ====================================================================

def pack_bits(q: "torch.Tensor", bits: int) -> "torch.Tensor":
    orig_device = q.device
    q_cpu = q.detach().to("cpu", dtype=torch.int64)
    n = q_cpu.numel()
    n_bytes = (n * bits + 7) // 8
    packed = torch.zeros(n_bytes, dtype=torch.int32, device="cpu")
    starts = torch.arange(n, dtype=torch.int64, device="cpu") * bits
    for b in range(bits):
        bitvals = (q_cpu >> b) & 1
        pos = starts + b
        byte_idx = pos // 8
        bit_idx = (pos % 8)
        set_mask = bitvals.bool()
        if set_mask.any():
            contrib = (1 << bit_idx[set_mask]).to(torch.int32)
            packed.index_put_((byte_idx[set_mask],), contrib, accumulate=True)
    return packed.to(torch.uint8).to(orig_device)


def unpack_bits(packed: "torch.Tensor", n: int, bits: int) -> "torch.Tensor":
    out = torch.zeros(n, dtype=torch.int64, device=packed.device)
    starts = torch.arange(n, dtype=torch.int64, device=packed.device) * bits
    for b in range(bits):
        pos = starts + b
        byte_idx = pos // 8
        bit_idx = (pos % 8).to(torch.uint8)
        bitval = (packed[byte_idx] >> bit_idx) & 1
        out |= (bitval.to(torch.int64) << b)
    return out


def quantize_blockwise(w: "torch.Tensor", bits: int, block_size: int = 128):
    orig_shape = w.shape
    orig_device = w.device
    flat = w.detach().to("cpu", dtype=torch.float32).reshape(-1)
    n = flat.numel()
    pad = (-n) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
    blocks = flat.reshape(-1, block_size)
    qmax = 2 ** bits - 1
    bmin = blocks.min(dim=1, keepdim=True).values
    bmax = blocks.max(dim=1, keepdim=True).values
    scale = ((bmax - bmin) / qmax).clamp_min(1e-8)
    zero_point = torch.round(-bmin / scale)
    q = torch.clamp(torch.round(blocks / scale + zero_point), 0, qmax)
    q_flat = q.reshape(-1)[:n].to(torch.int64)
    return q_flat.to(orig_device), scale.reshape(-1).to(orig_device), zero_point.reshape(-1).to(orig_device), n, orig_shape


def dequantize_blockwise(q_flat, scale, zero_point, n, orig_shape, bits, block_size, dtype, device):
    pad = (-n) % block_size
    q = q_flat
    if pad:
        q = torch.cat([q, torch.zeros(pad, dtype=q.dtype, device=q.device)])
    blocks = q.reshape(-1, block_size).float()
    w = (blocks - zero_point.unsqueeze(1)) * scale.unsqueeze(1)
    w = w.reshape(-1)[:n]
    return w.reshape(orig_shape).to(dtype=dtype, device=device)


class PackedLinear(nn.Module):
    """bits>=16: dense. 1<=bits<=8: bit-packed blockwise quant. bits==0: pruned/bypassed."""

    def __init__(self, orig: "nn.Linear", bits: int, block_size: int = 128):
        super().__init__()
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.bits = bits
        self.block_size = block_size
        self.has_bias = orig.bias is not None
        self.compute_dtype = orig.weight.dtype
        self.device = orig.weight.device
        self._cached_weight = None

        if bits == 0:
            self.packed = None
            self.weight_fp = None
            self.scale = None
            self.zero_point = None
        elif bits >= 16:
            self.register_buffer("weight_fp", orig.weight.data.clone())
            self.packed = None
        else:
            q_flat, scale, zero_point, n, shape = quantize_blockwise(orig.weight.data, bits, block_size)
            packed = pack_bits(q_flat, bits)
            self.register_buffer("packed", packed)
            self.register_buffer("scale", scale.to(torch.float32))
            self.register_buffer("zero_point", zero_point.to(torch.float32))
            self.orig_numel = n
            self.orig_shape = shape
            self.weight_fp = None

        if self.has_bias:
            self.register_buffer("bias", orig.bias.data.clone())
        else:
            self.bias = None

    def stored_bytes(self) -> int:
        if self.bits == 0:
            total = 0
        elif self.bits >= 16:
            total = self.weight_fp.numel() * self.weight_fp.element_size()
        else:
            total = self.packed.numel() * self.packed.element_size()
            total += self.scale.numel() * self.scale.element_size()
            total += self.zero_point.numel() * self.zero_point.element_size()
        if self.has_bias:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def dense_equivalent_bytes(self) -> int:
        numel = self.in_features * self.out_features
        total = numel * 2
        if self.has_bias:
            total += self.out_features * 2
        return total

    def _materialize_weight(self) -> Optional["torch.Tensor"]:
        if self.bits == 0:
            return None
        if self.bits >= 16:
            return self.weight_fp
        if self._cached_weight is not None:
            return self._cached_weight
        q_flat = unpack_bits(self.packed, self.orig_numel, self.bits)
        w = dequantize_blockwise(q_flat, self.scale, self.zero_point, self.orig_numel,
                                 self.orig_shape, self.bits, self.block_size,
                                 self.compute_dtype, self.packed.device)
        self._cached_weight = w
        return w

    def pre_materialize(self):
        if self.bits not in (0, 16):
            self._cached_weight = self._materialize_weight()

    def clear_materialized(self):
        if self.bits not in (0, 16):
            self._cached_weight = None

    def forward(self, x):
        if self.bits == 0:
            out_shape = list(x.shape[:-1]) + [self.out_features]
            if self.has_bias:
                return self.bias.expand(*out_shape).to(dtype=x.dtype)
            return torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        w = self._materialize_weight()
        return F.linear(x, w, self.bias)


def enumerate_linear_projections(model) -> List[Tuple[str, "nn.Module", str]]:
    refs = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            if not any(child_name.endswith(p) or child_name == p for p in PROJECTION_NAMES):
                continue
            full_name = f"{name}.{child_name}" if name else child_name
            refs.append((full_name, module, child_name))
    return refs


def layer_idx_from_name(name: str) -> int:
    for tok in name.split("."):
        if tok.isdigit():
            return int(tok)
    return -1


def proj_type_from_name(name: str) -> str:
    for p in PROJECTION_NAMES:
        if name.endswith(p):
            return p
    return "unknown"


def replace_with_packed_linear(model, bit_assignment: Dict[str, int], block_size: int = 128) -> Dict[str, dict]:
    refs = enumerate_linear_projections(model)
    audit = {}
    for full_name, parent, child_name in refs:
        bits = bit_assignment.get(full_name, 16)
        orig_linear = getattr(parent, child_name)
        packed = PackedLinear(orig_linear, bits=bits, block_size=block_size)
        setattr(parent, child_name, packed)
        stored = packed.stored_bytes()
        dense = packed.dense_equivalent_bytes()
        audit[full_name] = {
            "bits": bits,
            "layer_idx": layer_idx_from_name(full_name),
            "proj_type": proj_type_from_name(full_name),
            "stored_bytes": stored,
            "dense_fp16_equivalent_bytes": dense,
            "compression_ratio_vs_fp16": stored / max(dense, 1),
            "numel": orig_linear.in_features * orig_linear.out_features,
        }
    return audit


def swap_in_packed_linear(model, bit_assignment: Dict[str, int], block_size: int = 128) -> Dict[str, "nn.Linear"]:
    refs = enumerate_linear_projections(model)
    originals: Dict[str, "nn.Linear"] = {}
    for full_name, parent, child_name in refs:
        bits = bit_assignment.get(full_name, 16)
        orig_linear = getattr(parent, child_name)
        originals[full_name] = orig_linear
        packed = PackedLinear(orig_linear, bits=bits, block_size=block_size)
        setattr(parent, child_name, packed)
    return originals


def restore_original_linear(model, originals: Dict[str, "nn.Linear"]) -> None:
    for full_name, orig_linear in originals.items():
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        current = getattr(parent, child_name)
        setattr(parent, child_name, orig_linear)
        del current
    gc.collect()
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()


def pre_materialize_model_weights(model):
    for m in model.modules():
        if isinstance(m, PackedLinear):
            m.pre_materialize()


def clear_materialized_model_weights(model):
    for m in model.modules():
        if isinstance(m, PackedLinear):
            m.clear_materialized()


# ====================================================================
# 4. M3 — BoseComp (Fermi-Dirac Weight Allocation), task-specific
# (math unchanged from original v6 — independently re-verified logic)
# ====================================================================

@dataclass
class ProjectionRef:
    name: str
    layer_idx: int
    proj_type: str
    module: "nn.Module"


def enumerate_projections(model) -> List[ProjectionRef]:
    refs = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        matched = next((p for p in PROJECTION_NAMES if name.endswith(p)), None)
        if matched is None:
            continue
        refs.append(ProjectionRef(name=name, layer_idx=layer_idx_from_name(name), proj_type=matched, module=module))
    return refs


def _extract_lm_loss(model, out, input_ids) -> "torch.Tensor":
    if hasattr(out, "loss") and out.loss is not None:
        return out.loss
    logits = getattr(out, "logits", None)
    if logits is None and hasattr(out, "last_hidden_state"):
        if hasattr(model, "lm_head"):
            logits = model.lm_head(out.last_hidden_state)
        elif hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
            logits = model.get_output_embeddings()(out.last_hidden_state)
        else:
            raise RuntimeError("Model did not return logits or lm_head to compute loss.")
    if logits is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    raise RuntimeError("Cannot compute LM loss from output.")


def compute_jacobian_sensitivity(model, tokenizer, projections, calib_texts, device, max_len=512) -> Dict[str, float]:
    grad_accum = {p.name: 0.0 for p in projections}
    handles = []

    def make_hook(pname):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                grad_accum[pname] += grad_output[0].detach().float().norm(p=2).item()
        return hook

    for p in projections:
        handles.append(p.module.register_full_backward_hook(make_hook(p.name)))

    model.train()
    n = 0
    for text in calib_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            continue
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=input_ids)
        loss = _extract_lm_loss(model, out, input_ids)
        loss.backward()
        n += 1
    model.eval()
    for h in handles:
        h.remove()
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    n = max(n, 1)
    return {k: v / n for k, v in grad_accum.items()}


def _calib_loss(model, tokenizer, texts, device, max_len) -> float:
    losses = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            continue
        out = model(input_ids=input_ids, labels=input_ids)
        loss = _extract_lm_loss(model, out, input_ids)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def compute_causal_ablation_sensitivity(model, tokenizer, projections, calib_texts, device, max_len=512) -> Dict[str, float]:
    with torch.no_grad():
        baseline_loss = _calib_loss(model, tokenizer, calib_texts, device, max_len)
    D = {}
    for p in projections:
        w = p.module.weight
        original = w.data.clone()
        w.data.zero_()
        with torch.no_grad():
            ablated_loss = _calib_loss(model, tokenizer, calib_texts, device, max_len)
        w.data.copy_(original)
        D[p.name] = ablated_loss - baseline_loss
    return D


def _minmax_normalize(d: Dict[str, float]) -> Dict[str, float]:
    vals = np.array(list(d.values()), dtype=np.float64)
    lo, hi = vals.min(), vals.max()
    span = max(hi - lo, 1e-12)
    return {k: (v - lo) / span for k, v in d.items()}


def compute_sensitivity_spectrum(J: Dict[str, float], D: Dict[str, float], cfg: BoseCompConfig):
    J_norm, D_norm = _minmax_normalize(J), _minmax_normalize(D)
    names = list(J.keys())
    S = np.array([cfg.alpha * J_norm[n] + cfg.beta * D_norm[n] for n in names])
    epsilon = 1.0 - S
    order = np.argsort(epsilon)
    sorted_names = [names[i] for i in order]
    return sorted_names, epsilon[order], J_norm, D_norm


def spectral_gap_cut(epsilon_sorted: np.ndarray, cfg: BoseCompConfig) -> int:
    M = len(epsilon_sorted)
    k_max = max(1, int(cfg.k_max_frac * M))
    diffs = np.diff(epsilon_sorted[: k_max + 1])
    return int(np.argmax(diffs))


def fermi_occupation(epsilon: np.ndarray, mu: float, T: float) -> np.ndarray:
    x = np.clip((epsilon - mu) / max(T, 1e-8), -50, 50)
    return 1.0 / (np.exp(x) + 1.0)


def continuous_bits(epsilon_thermal: np.ndarray, T: float, mu: float, b_max: int) -> np.ndarray:
    return b_max * fermi_occupation(epsilon_thermal, mu, T)


def project_to_lattice(b_continuous: np.ndarray, lattice: Tuple[int, ...]) -> np.ndarray:
    lattice_arr = np.array(lattice)
    return np.array([lattice_arr[np.argmin(np.abs(lattice_arr - b))] for b in b_continuous])


def cooling_sweep(model, tokenizer, device, sorted_names, epsilon_sorted, k_star,
                  projections_by_name, calib_texts, max_len, cfg: BoseCompConfig,
                  block_size: int) -> Tuple[Dict[str, int], List[dict]]:
    core_names = sorted_names[: k_star + 1]
    thermal_names = sorted_names[k_star + 1:]
    thermal_eps = epsilon_sorted[k_star + 1:]

    with torch.no_grad():
        baseline_loss = _calib_loss(model, tokenizer, calib_texts, device, max_len)

    thermal_eps_sorted = np.sort(thermal_eps) if len(thermal_eps) else np.array([0.0])
    eps0 = thermal_eps_sorted[0] if len(thermal_eps_sorted) else 0.0
    span = max(thermal_eps_sorted[-1] - eps0, 1e-6) if len(thermal_eps_sorted) > 1 else 1e-6
    T0 = cfg.t0_frac * span

    best_assignment = {n: cfg.b_core for n in core_names}
    best_assignment.update({n: cfg.b_max for n in thermal_names})

    cooling_log = []
    stop_reason = "step_budget_exhausted"

    def trial_eval(assignment: Dict[str, int]) -> float:
        originals = {}
        for name, bits in assignment.items():
            module = projections_by_name[name].module
            originals[name] = module.weight.data.clone()
            if bits == 0:
                module.weight.data.zero_()
            elif bits < 16:
                q_flat, scale, zp, n, shape = quantize_blockwise(module.weight.data, bits, block_size)
                module.weight.data.copy_(
                    dequantize_blockwise(q_flat, scale, zp, n, shape, bits, block_size,
                                         module.weight.dtype, module.weight.device))
        with torch.no_grad():
            loss = _calib_loss(model, tokenizer, calib_texts, device, max_len)
        for name, orig in originals.items():
            projections_by_name[name].module.weight.data.copy_(orig)
        return loss

    attn_targets = ("q_proj", "k_proj", "v_proj", "o_proj")

    for step in range(cfg.max_cooling_steps):
        frac = step / max(cfg.max_cooling_steps - 1, 1)
        T = T0 * (cfg.cooling_rate ** step)
        mu = eps0 + (cfg.mu_start_frac - (cfg.mu_start_frac - cfg.mu_end_frac) * frac) * span

        if len(thermal_names) == 0:
            stop_reason = "no_thermal_projections"
            break

        b_cont = continuous_bits(thermal_eps, T, mu, cfg.b_max)
        b_quant = project_to_lattice(b_cont, cfg.bit_lattice)

        for i, name in enumerate(thermal_names):
            if proj_type_from_name(name) in attn_targets:
                b_quant[i] = max(b_quant[i], cfg.min_attn_bits)

        assignment = {n: cfg.b_core for n in core_names}
        assignment.update({n: int(b) for n, b in zip(thermal_names, b_quant)})

        if step == 0:
            anchor_assignment = {n: cfg.b_core for n in core_names}
            anchor_assignment.update({n: cfg.b_max for n in thermal_names})
            anchor_loss = trial_eval(anchor_assignment)
            schedule_loss = trial_eval(assignment)

            cooling_log.append({
                "step": step, "T": float(T), "mu": float(mu),
                "trial_loss": float(schedule_loss),
                "anchor_loss": float(anchor_loss),
                "baseline_loss": float(baseline_loss),
                "mean_thermal_bits": float(np.mean(b_quant)) if len(b_quant) else 0.0,
            })
            logger.info(
                "Cooling step 0: T=%.5f mu=%.5f anchor_loss=%.5f schedule_loss=%.5f "
                "(baseline %.5f) mean_thermal_bits=%.2f",
                T, mu, anchor_loss, schedule_loss, baseline_loss,
                float(np.mean(b_quant)) if len(b_quant) else 0.0)

            if schedule_loss <= baseline_loss + cfg.loss_tolerance:
                best_assignment = assignment
            elif anchor_loss <= baseline_loss + cfg.loss_tolerance:
                best_assignment = anchor_assignment
                logger.info("Step 0: schedule assignment exceeded tolerance, "
                            "falling back to uniform-b_max anchor (still accepted).")
            else:
                best_assignment = anchor_assignment
                stop_reason = "loss_tolerance_too_tight_even_for_anchor_at_step_0"
                logger.warning("Step 0: even uniform b_max=%d exceeded tolerance. Stopping.", cfg.b_max)
                break
            continue

        trial_loss = trial_eval(assignment)
        cooling_log.append({
            "step": step, "T": float(T), "mu": float(mu),
            "trial_loss": float(trial_loss), "baseline_loss": float(baseline_loss),
            "mean_thermal_bits": float(np.mean(b_quant)) if len(b_quant) else 0.0,
        })
        logger.info("Cooling step %d: T=%.5f mu=%.5f trial_loss=%.5f (baseline %.5f) mean_thermal_bits=%.2f",
                    step, T, mu, trial_loss, baseline_loss,
                    float(np.mean(b_quant)) if len(b_quant) else 0.0)

        if trial_loss > baseline_loss + cfg.loss_tolerance:
            stop_reason = f"loss_tolerance_exceeded_at_step_{step}"
            logger.info("Cooling stopped: %s (trial_loss=%.5f > baseline+tol=%.5f).",
                        stop_reason, trial_loss, baseline_loss + cfg.loss_tolerance)
            break

        best_assignment = assignment
    else:
        logger.info("Cooling stopped: %s after %d steps.", stop_reason, cfg.max_cooling_steps)

    return best_assignment, cooling_log


def run_bosecomp(model, tokenizer, device, calib_texts, max_len, cfg: BoseCompConfig) -> Tuple[Dict[str, int], dict]:
    projections = enumerate_projections(model)
    projections_by_name = {p.name: p for p in projections}
    logger.info("Enumerated %d projection matrices", len(projections))

    J = compute_jacobian_sensitivity(model, tokenizer, projections, calib_texts, device, max_len)
    D = compute_causal_ablation_sensitivity(model, tokenizer, projections, calib_texts, device, max_len)
    sorted_names, epsilon_sorted, J_norm, D_norm = compute_sensitivity_spectrum(J, D, cfg)
    k_star = spectral_gap_cut(epsilon_sorted, cfg)
    logger.info("Spectral gap cut k*=%d/%d (core size=%d)", k_star, len(sorted_names), k_star + 1)

    assignment, cooling_log = cooling_sweep(
        model, tokenizer, device, sorted_names, epsilon_sorted, k_star,
        projections_by_name, calib_texts, max_len, cfg, cfg.quant_block_size)

    core_set = set(sorted_names[: k_star + 1])
    eps_by_name = {n: float(e) for n, e in zip(sorted_names, epsilon_sorted)}
    per_projection_table = []
    for p in projections:
        per_projection_table.append({
            "name": p.name, "layer_idx": p.layer_idx, "proj_type": p.proj_type,
            "J_raw": float(J[p.name]), "D_raw": float(D[p.name]),
            "J_norm": float(J_norm[p.name]), "D_norm": float(D_norm[p.name]),
            "epsilon": eps_by_name[p.name],
            "membership": "condensate_core" if p.name in core_set else "thermal_cloud",
            "assigned_bits": assignment[p.name],
        })
    per_projection_table.sort(key=lambda r: (r["layer_idx"], r["proj_type"]))

    thermal_bits_list = [row["assigned_bits"] for row in per_projection_table if row["membership"] == "thermal_cloud"]
    bit_histogram = {}
    for b in thermal_bits_list:
        bit_histogram[str(b)] = bit_histogram.get(str(b), 0) + 1

    diagnostics = {
        "k_star": k_star, "core_size": k_star + 1,
        "n_projections_total": len(projections),
        "n_thermal": len(projections) - (k_star + 1),
        "cooling_log": cooling_log,
        "final_cooling_steps_used": len(cooling_log),
        "thermal_bit_histogram": bit_histogram,
        "per_projection": per_projection_table,
    }
    return assignment, diagnostics


def bit_assignment_summary_from_bosecomp(bit_assignment: Dict[str, int], bose_diag: dict) -> dict:
    thermal_bits = [b for b in bit_assignment.values() if b < 16]
    return {
        "n_core_16bit": sum(1 for b in bit_assignment.values() if b >= 16),
        "n_thermal": len(thermal_bits),
        "n_pruned_0bit": sum(1 for b in thermal_bits if b == 0),
        "mean_thermal_bits": (sum(thermal_bits) / len(thermal_bits)) if thermal_bits else 0,
        "min_thermal_bits": min(thermal_bits) if thermal_bits else None,
        "max_thermal_bits": max(thermal_bits) if thermal_bits else None,
        "thermal_bit_histogram": bose_diag.get("thermal_bit_histogram", {}),
        "cooling_stop_step": len(bose_diag.get("cooling_log", [])),
    }


def run_m3_task_specific(model, tokenizer, device, run_cfg: "RunConfig") -> Tuple[dict, dict]:
    results: Dict[str, dict] = {}
    audit: Dict[str, dict] = {}

    for task_name, task_fn in TASK_REGISTRY.items():
        logger.info("M3 (task-specific BoseComp): starting task=%s", task_name)
        task_calib_texts = build_task_calibration_texts(task_name, run_cfg.task_n_calib_per_task)

        bit_assignment, bose_diag = run_bosecomp(
            model, tokenizer, device, task_calib_texts, run_cfg.calib_seq_len, run_cfg.bosecomp)

        originals = swap_in_packed_linear(model, bit_assignment, block_size=run_cfg.bosecomp.quant_block_size)
        try:
            weights_gb = model_weight_footprint_gb(model)
            weights_fp16_equiv_gb = model_weight_footprint_fp16_equivalent_gb(model)
            compression = compute_compression_metrics(model, None)
            bit_summary = bit_assignment_summary_from_bosecomp(bit_assignment, bose_diag)

            pre_materialize_model_weights(model)
            task_metrics = task_fn(model, tokenizer, device)

            results[task_name] = {
                "task_metrics": task_metrics,
                "weights_gb": weights_gb,
                "weights_fp16_equivalent_gb": weights_fp16_equiv_gb,
                "compression": compression,
                "bit_assignment_summary": bit_summary,
                "calib_n_samples": len(task_calib_texts),
            }
            audit[task_name] = bose_diag
        finally:
            restore_original_linear(model, originals)
            logger.info("M3: restored original weights after task=%s, freed packed buffers.", task_name)

    return results, audit


# ====================================================================
# 4c. BOSESIR — BOSE-Condense
# BUG FIX (v8): gate tensor now created directly in the model's real
# compute dtype so it stays the leaf tensor in the autograd graph. See
# module docstring "BUG FIX 1" for the full explanation of why the old
# torch.float32-then-cast approach silently zeroed gradients under
# bfloat16 model weights.
# ====================================================================

def _get_model_config_attr(config, keys: List[str], default=None):
    for cfg in [config, getattr(config, "text_config", None)]:
        if cfg is None:
            continue
        for key in keys:
            if hasattr(cfg, key):
                val = getattr(cfg, key)
                if val is not None:
                    return val
    return default


def _get_model_geometry(model):
    cfg = model.config
    hidden_size = _get_model_config_attr(cfg, ["hidden_size", "d_model", "dim", "embed_dim"])
    num_layers = _get_model_config_attr(cfg, ["num_hidden_layers", "num_layers", "n_layers", "n_layer"])
    num_heads = _get_model_config_attr(cfg, ["num_attention_heads", "num_heads", "n_head", "n_heads"], default=1)

    if num_layers is None:
        try:
            num_layers = len(_get_transformer_layers(model))
        except Exception:
            num_layers = 32

    if hidden_size is None:
        for p in model.parameters():
            if p.ndim == 2:
                hidden_size = p.shape[-1]
                break
        hidden_size = hidden_size or 4096

    return int(hidden_size), int(num_layers), int(num_heads)


def _get_transformer_layers(model) -> List["nn.Module"]:
    candidate_paths = [
        ("model", "layers"), ("model", "model", "layers"), ("model", "decoder", "layers"),
        ("transformer", "layers"), ("transformer", "h"), ("transformer", "blocks"),
        ("decoder", "layers"), ("decoder", "block"), ("layers",), ("h",), ("blocks",),
        ("language_model", "model", "layers"), ("model", "language_model", "layers"),
        ("backbone", "layers"),
    ]
    for path in candidate_paths:
        obj = model
        for step in path:
            obj = getattr(obj, step, None)
            if obj is None:
                break
        if obj is not None and isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
            return list(obj)
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and any(sub in name.lower() for sub in ["layers", "blocks", ".h."]):
            return list(module)
    raise RuntimeError("Could not locate transformer decoder layer list.")


class _LayerGateHook:
    """
    Multiplicative gate applied to a decoder layer's down_proj output, used
    to measure per-layer utility via the gradient the loss places on the
    gate. FIXED in v8: `make_gate` now REQUIRES an explicit `dtype` — the
    caller (`_register_layer_gates` / `bose_evidence_loop`) passes the
    model's actual parameter dtype (e.g. torch.bfloat16), so `self.gate`
    IS the tensor multiplied directly against the (same-dtype) layer
    output, with no intermediate `.to(dtype=...)` cast node breaking the
    autograd leaf. Previously `self.gate` was always float32 and got cast
    to the activation dtype inside the hook, which made `self.gate.grad`
    silently come back None whenever the model ran in bf16/fp16 (see
    module docstring "BUG FIX 1").
    """
    def __init__(self):
        self.gate: Optional["torch.Tensor"] = None
        self.last_output_detached: Optional["torch.Tensor"] = None

    def make_gate(self, device, dtype):
        # dtype MUST match the activation dtype this gate will be multiplied
        # against, so self.gate itself (not a cast of it) is the tensor that
        # participates in the forward multiply and receives the gradient.
        self.gate = torch.ones((), device=device, dtype=dtype, requires_grad=True)
        return self.gate

    def hook(self, module, inputs, output):
        # No dtype cast here anymore — self.gate is already output's dtype,
        # so this multiply keeps self.gate as the leaf that grad lands on.
        gated = output * self.gate
        self.last_output_detached = output.detach()
        return gated


def _register_layer_gates(model, layers: List["nn.Module"]) -> Dict[int, _LayerGateHook]:
    hooks: Dict[int, _LayerGateHook] = {}
    for layer_idx, layer in enumerate(layers):
        down = None
        for name, child in layer.named_modules():
            if name.endswith("down_proj") and isinstance(child, nn.Linear):
                down = child
                break
        if down is None:
            continue
        h = _LayerGateHook()
        h.make_gate(device=down.weight.device, dtype=down.weight.dtype)
        down.register_forward_hook(h.hook)
        hooks[layer_idx] = h
    return hooks


def _assert_gate_grads_nonzero(u_raw: Dict[int, float], step: int) -> None:
    """
    Guards against a repeat of the v7 gate-dtype bug (or any future bug
    with the same symptom): if EVERY layer's captured utility signal is
    exactly 0.0 for an entire evidence step, that's not a real evidence
    signal — it means gradients never reached the gates at all. Raise
    loudly instead of silently continuing with degenerate data.
    """
    if u_raw and all(v == 0.0 for v in u_raw.values()):
        raise EvidenceSignalDegenerateError(
            f"BOSESIR evidence step {step}: every layer's captured gate "
            f"gradient magnitude was exactly 0.0. This previously happened "
            f"silently due to a gate/activation dtype mismatch breaking the "
            f"autograd leaf (see module docstring BUG FIX 1) — that specific "
            f"bug is fixed in this revision, but this assertion stays as a "
            f"guard in case of a similar future regression. If you see this "
            f"error, check that gate.dtype matches the hooked layer's output "
            f"dtype and that loss.backward() is actually reaching the hooked "
            f"module."
        )


def bose_evidence_loop(model, tokenizer, device, calib_texts: List[str], layers: List["nn.Module"],
                        cfg: BoseSirConfig) -> Dict[int, dict]:
    if not calib_texts:
        raise CalibrationDataUnavailableError(
            "bose_evidence_loop received an empty calib_texts list. This script "
            "does not substitute a fabricated placeholder sentence — fix the "
            "upstream calibration data source."
        )
    n_layers = len(layers)
    U = {l: 0.0 for l in range(n_layers)}
    R = {l: 0.0 for l in range(n_layers)}
    I = {l: 0.0 for l in range(n_layers)}
    N = {l: 0.0 for l in range(n_layers)}
    D = {l: 0.0 for l in range(n_layers)}
    u_bar_prev = {l: 0.0 for l in range(n_layers)}
    running_mean_act: Dict[int, Optional["torch.Tensor"]] = {l: None for l in range(n_layers)}

    hooks = _register_layer_gates(model, layers)
    # Model's real compute dtype — passed to make_gate on every re-arm below
    # so the gate tensor always matches the activation dtype it multiplies.
    model_param_dtype = next(model.parameters()).dtype

    steps = cfg.n_evidence_steps
    texts_per_step = max(1, len(calib_texts) // steps)

    model.train()
    for step in range(steps):
        batch_texts = calib_texts[step * texts_per_step:(step + 1) * texts_per_step] or calib_texts[:1]

        for h in hooks.values():
            h.make_gate(device=next(model.parameters()).device, dtype=model_param_dtype)

        step_losses = []
        step_saw_any_grad = False
        for text in batch_texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=cfg.max_len).to(device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue
            model.zero_grad(set_to_none=True)
            for h in hooks.values():
                if h.gate.grad is not None:
                    h.gate.grad = None
            out = model(input_ids=input_ids, labels=input_ids)
            loss = _extract_lm_loss(model, out, input_ids)
            loss.backward()
            step_losses.append(loss.item())

            u_raw = {}
            for l, h in hooks.items():
                if h.gate.grad is not None:
                    u_val = float(-h.gate.detach().item() * h.gate.grad.item())
                else:
                    u_val = 0.0
                u_raw[l] = max(0.0, u_val)
            if any(v != 0.0 for v in u_raw.values()):
                step_saw_any_grad = True

            denom = sum(u_raw.values()) + cfg.eps0
            u_tilde = {l: u_raw[l] / denom for l in u_raw}

            for l in range(n_layers):
                u_l = u_raw.get(l, 0.0)
                ut_l = u_tilde.get(l, 0.0)
                U[l] = cfg.rho_u * U[l] + (1 - cfg.rho_u) * u_l
                R_prev = R[l]
                R[l] = cfg.rho_r * R_prev + (1 - cfg.rho_r) * ut_l
                I[l] = cfg.rho_i * I[l] + (1 - cfg.rho_i) * (ut_l - u_bar_prev[l]) ** 2
                novelty_raw = min(ut_l / (R_prev + cfg.eps0), cfg.novelty_clip_max)
                N[l] = cfg.rho_n * N[l] + (1 - cfg.rho_n) * novelty_raw
                u_bar_prev[l] = ut_l

                h = hooks.get(l)
                if h is not None and h.last_output_detached is not None:
                    m_l = h.last_output_detached.detach().float().mean(dim=tuple(range(h.last_output_detached.dim() - 1)))
                    prev_mean = running_mean_act[l]
                    if prev_mean is None:
                        cos_sim = 1.0
                        running_mean_act[l] = m_l
                    else:
                        denom_cos = (m_l.norm() * prev_mean.norm()).clamp_min(cfg.eps0)
                        cos_sim = float((m_l @ prev_mean) / denom_cos)
                        running_mean_act[l] = 0.9 * prev_mean + 0.1 * m_l
                    D[l] = cfg.rho_d * D[l] + (1 - cfg.rho_d) * (1.0 - cos_sim)

        # Guard against a silent repeat of the gate-dtype bug: if this
        # entire step produced zero gradient signal on every layer for
        # every text, something upstream broke the autograd path again.
        if batch_texts and not step_saw_any_grad:
            _assert_gate_grads_nonzero({0: 0.0}, step)

        logger.info("BOSESIR evidence step %d/%d: mean_loss=%.5f", step + 1, steps,
                    float(np.mean(step_losses)) if step_losses else float("nan"))

    model.eval()
    model.zero_grad(set_to_none=True)
    hooks.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return {l: {"U": U[l], "R": R[l], "I": I[l], "N": N[l], "D": D[l]} for l in range(n_layers)}


def bose_energy_from_evidence(evidence_by_layer: Dict[int, dict], cfg: BoseSirConfig) -> Dict[int, float]:
    layer_idxs = sorted(evidence_by_layer.keys())

    def z(key: str) -> Dict[int, float]:
        vals = np.array([evidence_by_layer[l][key] for l in layer_idxs], dtype=np.float64)
        mean, var = vals.mean(), vals.var()
        z_vals = (vals - mean) / np.sqrt(var + 1e-8)
        return {l: float(v) for l, v in zip(layer_idxs, z_vals)}

    I_hat, N_hat, D_hat, R_hat, U_hat = z("I"), z("N"), z("D"), z("R"), z("U")

    ablation = cfg.energy_ablation.upper()
    E = {}
    for l in layer_idxs:
        if ablation == "A1":
            e = -cfg.delta * U_hat[l]
        elif ablation == "A2":
            e = -cfg.delta * U_hat[l] - cfg.gamma * R_hat[l]
        elif ablation == "A3":
            e = cfg.alpha * I_hat[l] - cfg.gamma * R_hat[l] - cfg.delta * U_hat[l]
        else:
            e = (cfg.alpha * I_hat[l] + cfg.beta * N_hat[l] + cfg.chi * D_hat[l]
                 - cfg.gamma * R_hat[l] - cfg.delta * U_hat[l])
        E[l] = float(e)
    return E


def solve_shared_mu_bisection(E_by_layer: Dict[int, float], c_l_by_layer: Dict[int, int],
                               p_l_by_layer: Dict[int, int], N_F: float, T_B: float,
                               pole_eps: float, iters: int, margin: float) -> Tuple[float, Dict[int, float]]:
    layer_idxs = sorted(E_by_layer.keys())
    E = {l: E_by_layer[l] for l in layer_idxs}
    c = {l: c_l_by_layer[l] for l in layer_idxs}
    p = {l: p_l_by_layer[l] for l in layer_idxs}

    fixed: Dict[int, float] = {}
    free = set(layer_idxs)
    mu: Optional[float] = None

    for _round in range(len(layer_idxs) + 1):
        if not free:
            break
        residual_budget = N_F - sum(fixed.values())
        free_list = sorted(free)
        E_free = np.array([E[l] for l in free_list])
        c_free = np.array([c[l] for l in free_list], dtype=np.float64)
        p_free = np.array([p[l] for l in free_list], dtype=np.float64)

        if residual_budget <= p_free.sum() + 1e-9:
            for l in free_list:
                fixed[l] = p[l]
            free = set()
            break

        if residual_budget >= c_free.sum() - 1e-9:
            for l in free_list:
                fixed[l] = c[l]
            free = set()
            break

        e_min = E_free.min()

        def n_of_mu(mu_val: float) -> np.ndarray:
            x = (E_free - mu_val) / max(T_B, 1e-8)
            return 1.0 / (np.exp(np.clip(x, -50, 50)) - 1.0 + pole_eps)

        def f_gap(gap: float) -> float:
            occ = np.clip(n_of_mu(e_min - gap), p_free, c_free)
            return float(occ.sum() - residual_budget)

        gap_lo, gap_hi = 1e-12, margin
        f_lo, f_hi = f_gap(gap_lo), f_gap(gap_hi)

        if f_lo < 0:
            occ_at_pole = np.clip(n_of_mu(e_min - gap_lo), p_free, c_free)
            saturated_mask = occ_at_pole >= c_free - 1e-9
            if not saturated_mask.any():
                for l in free_list:
                    fixed[l] = c[l]
                free = set()
                break
            for i, l in enumerate(free_list):
                if saturated_mask[i]:
                    fixed[l] = c[l]
                    free.discard(l)
            continue

        if f_hi > 0:
            for l in free_list:
                fixed[l] = p[l]
            free = set()
            break

        lo_gap, hi_gap = gap_lo, gap_hi
        for _ in range(iters):
            mid_gap = float(np.sqrt(lo_gap * hi_gap))
            if f_gap(mid_gap) > 0:
                lo_gap = mid_gap
            else:
                hi_gap = mid_gap
        gap = float(np.sqrt(lo_gap * hi_gap))
        mu = e_min - gap
        occ = np.clip(n_of_mu(mu), p_free, c_free)
        for i, l in enumerate(free_list):
            fixed[l] = float(occ[i])
        free = set()

    for l in free:
        fixed[l] = p[l]

    n_continuous = {l: fixed[l] for l in layer_idxs}
    return (mu if mu is not None else float("nan")), n_continuous


def quanta_to_bit_assignment(n_by_layer: Dict[int, float], core_layers: set,
                              layers_module_names: Dict[int, List[str]],
                              min_attn_bits: int = 4) -> Dict[str, int]:
    bit_assignment: Dict[str, int] = {}
    attn_types = ("q_proj", "k_proj", "v_proj", "o_proj")
    for layer_idx, proj_names in layers_module_names.items():
        if layer_idx in core_layers:
            bits = 16
        else:
            n_cont = n_by_layer.get(layer_idx, float(BOSESIR_C_L))
            n_int = int(round(np.clip(n_cont, 0, BOSESIR_C_L)))
            bits = QUANTA_TO_BITS[n_int]
        for name in proj_names:
            proj_bits = bits
            if layer_idx not in core_layers and proj_type_from_name(name) in attn_types:
                proj_bits = max(proj_bits, min_attn_bits)
            bit_assignment[name] = proj_bits
    return bit_assignment


def run_bosesir(model, tokenizer, device, run_cfg: "RunConfig") -> Tuple[dict, dict]:
    results: Dict[str, dict] = {}
    audit: Dict[str, dict] = {}

    layers = _get_transformer_layers(model)
    n_layers = len(layers)
    n_core = max(1, int(run_cfg.bosesir.core_layer_frac * n_layers))
    core_layers = set(range(n_core))

    refs = enumerate_linear_projections(model)
    layers_module_names: Dict[int, List[str]] = {l: [] for l in range(n_layers)}
    for full_name, _parent, _child in refs:
        lidx = layer_idx_from_name(full_name)
        if lidx in layers_module_names:
            layers_module_names[lidx].append(full_name)

    c_l_by_layer = {l: len(layers_module_names[l]) for l in range(n_layers)}
    p_l_by_layer = {l: (c_l_by_layer[l] if l in core_layers else 2) for l in range(n_layers)}

    non_core_capacity = sum(c_l_by_layer[l] for l in range(n_layers) if l not in core_layers)
    N_F = run_cfg.bosesir.n_f_frac * non_core_capacity

    for task_name, task_fn in TASK_REGISTRY.items():
        logger.info("BOSESIR: starting task-specific BOSE-Condense allocation for task=%s", task_name)
        task_calib_texts = build_task_calibration_texts(task_name, run_cfg.bosesir_n_calib())

        evidence_by_layer = bose_evidence_loop(model, tokenizer, device, task_calib_texts, layers, run_cfg.bosesir)
        E_by_layer = bose_energy_from_evidence(evidence_by_layer, run_cfg.bosesir)

        noncore_E = {l: E_by_layer[l] for l in E_by_layer if l not in core_layers}
        noncore_c = {l: c_l_by_layer[l] for l in noncore_E}
        noncore_p = {l: p_l_by_layer[l] for l in noncore_E}

        mu, n_continuous_noncore = solve_shared_mu_bisection(
            noncore_E, noncore_c, noncore_p, N_F, run_cfg.bosesir.T_B,
            run_cfg.bosesir.pole_eps, run_cfg.bosesir.mu_bisect_iters, run_cfg.bosesir.mu_search_margin)

        n_by_layer = dict(n_continuous_noncore)
        for l in core_layers:
            n_by_layer[l] = float(c_l_by_layer[l])

        bit_assignment = quanta_to_bit_assignment(
            n_by_layer, core_layers, layers_module_names,
            min_attn_bits=run_cfg.bosesir.min_attn_bits)

        originals = swap_in_packed_linear(model, bit_assignment, block_size=run_cfg.bosesir.quant_block_size)
        try:
            weights_gb = model_weight_footprint_gb(model)
            weights_fp16_equiv_gb = model_weight_footprint_fp16_equivalent_gb(model)
            compression = compute_compression_metrics(model, None)

            noncore_bits = [bit_assignment[name] for l in range(n_layers) if l not in core_layers
                            for name in layers_module_names[l]]
            bit_summary = {
                "n_core_16bit_layers": len(core_layers),
                "n_thermal_layers": n_layers - len(core_layers),
                "n_pruned_0bit_layers": sum(1 for b in noncore_bits if b == 0),
                "mean_thermal_bits": (sum(noncore_bits) / len(noncore_bits)) if noncore_bits else 0,
                "min_thermal_bits": min(noncore_bits) if noncore_bits else None,
                "max_thermal_bits": max(noncore_bits) if noncore_bits else None,
                "mu": mu, "N_F": N_F, "T_B": run_cfg.bosesir.T_B,
                "energy_ablation": run_cfg.bosesir.energy_ablation,
            }

            pre_materialize_model_weights(model)
            task_metrics = task_fn(model, tokenizer, device)

            results[task_name] = {
                "task_metrics": task_metrics,
                "weights_gb": weights_gb,
                "weights_fp16_equivalent_gb": weights_fp16_equiv_gb,
                "compression": compression,
                "bit_assignment_summary": bit_summary,
                "calib_n_samples": len(task_calib_texts),
            }

            per_layer_audit = []
            for l in range(n_layers):
                n_val = n_by_layer[l]
                n_int = int(round(np.clip(n_val, 0, BOSESIR_C_L)))
                per_layer_audit.append({
                    "layer_idx": l, "is_core_layer": l in core_layers,
                    "evidence": evidence_by_layer.get(l, {}),
                    "energy_E_l": E_by_layer.get(l),
                    "n_l_continuous": n_val, "n_l_quanta": n_int,
                    "c_l": c_l_by_layer[l], "p_l": p_l_by_layer[l],
                    "assigned_bits_mlp": QUANTA_TO_BITS[n_int] if l not in core_layers else 16,
                    "assigned_bits_per_proj": {name: bit_assignment[name] for name in layers_module_names[l]},
                })
            audit[task_name] = {
                "mu": mu, "N_F": N_F, "T_B": run_cfg.bosesir.T_B,
                "energy_ablation": run_cfg.bosesir.energy_ablation,
                "core_layers": sorted(core_layers), "per_layer": per_layer_audit,
            }
        finally:
            restore_original_linear(model, originals)
            logger.info("BOSESIR: restored original weights after task=%s, freed packed buffers.", task_name)

    return results, audit


# ====================================================================
# 5. MEMORY / COMPRESSION METRICS (weight-only; latency/KV/energy moved
#    to kvcache_bench.py since they're primarily about the KV cache and
#    generation loop, not weight storage)
# ====================================================================

def model_weight_footprint_gb(model) -> float:
    total_bytes = 0
    packed_module_ids = set()
    for module in model.modules():
        if isinstance(module, PackedLinear):
            total_bytes += module.stored_bytes()
            packed_module_ids.add(id(module))
    for name, module in model.named_modules():
        if id(module) in packed_module_ids:
            continue
        for p in module.parameters(recurse=False):
            total_bytes += p.numel() * p.element_size()
        for b in module.buffers(recurse=False):
            total_bytes += b.numel() * b.element_size()
    return total_bytes / 1e9


def model_weight_footprint_fp16_equivalent_gb(model) -> float:
    total_bytes = 0
    for module in model.modules():
        if isinstance(module, PackedLinear):
            total_bytes += module.dense_equivalent_bytes()
    for name, module in model.named_modules():
        if isinstance(module, PackedLinear):
            continue
        for p in module.parameters(recurse=False):
            total_bytes += p.numel() * 2
        for b in module.buffers(recurse=False):
            total_bytes += b.numel() * 2
    return total_bytes / 1e9


def compute_compression_metrics(model, m0_weights_gb: Optional[float]) -> dict:
    actual_gb = model_weight_footprint_gb(model)
    fp16_equiv_gb = model_weight_footprint_fp16_equivalent_gb(model)
    metrics = {
        "actual_stored_weights_gb": actual_gb,
        "fp16_equivalent_weights_gb": fp16_equiv_gb,
        "compression_ratio_vs_fp16_equivalent": actual_gb / max(fp16_equiv_gb, 1e-9),
        "space_saved_pct_vs_fp16_equivalent": 100.0 * (1.0 - actual_gb / max(fp16_equiv_gb, 1e-9)),
        "effective_avg_bits_per_weight": (actual_gb / max(fp16_equiv_gb, 1e-9)) * 16.0,
    }
    if m0_weights_gb:
        metrics["compression_ratio_vs_measured_m0"] = actual_gb / max(m0_weights_gb, 1e-9)
        metrics["space_saved_pct_vs_measured_m0"] = 100.0 * (1.0 - actual_gb / max(m0_weights_gb, 1e-9))
    return metrics


# ====================================================================
# 6. DOWNSTREAM TASK EVALUATORS (real datasets; ICD-10 explicitly labeled
#    as a synthetic probe, never a benchmark, per module docstring)
# ====================================================================

def eval_sst2(model, tokenizer, device, n_samples: int = N_SST2) -> Dict[str, float]:
    """Sentiment classification. Dataset: stanfordnlp/sst2 (validation split)."""
    _require_datasets_available()
    with torch.no_grad():
        ds = load_dataset("stanfordnlp/sst2", split=f"validation[:{n_samples}]")
        pos_id = tokenizer.encode(" positive", add_special_tokens=False)[-1]
        neg_id = tokenizer.encode(" negative", add_special_tokens=False)[-1]
        correct = total = 0
        for ex in ds:
            prompt = f"Review: {ex['sentence'].strip()}\nSentiment (positive/negative):"
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            out = model(**enc)
            logits = getattr(out, "logits", None)
            if logits is None and hasattr(out, "last_hidden_state") and hasattr(model, "lm_head"):
                logits = model.lm_head(out.last_hidden_state)
            last_token_logits = logits[0, -1, :]
            pred = 1 if last_token_logits[pos_id] > last_token_logits[neg_id] else 0
            correct += int(pred == ex["label"])
            total += 1
    return {"sst2_acc": 100.0 * correct / max(total, 1)}


def eval_squad_v2(model, tokenizer, device, n_samples: int = N_SQUAD, max_new_tokens=32) -> Dict[str, float]:
    """Extractive QA / unanswerable detection. Dataset: rajpurkar/squad_v2 (validation split)."""
    _require_datasets_available()
    ds = load_dataset("rajpurkar/squad_v2", split=f"validation[:{n_samples}]")
    metric = hf_evaluate.load("squad_v2") if _EVALUATE_AVAILABLE else None
    predictions, references = [], []
    with torch.no_grad():
        for i, ex in enumerate(ds):
            prompt = (f"Context: {ex['context']}\nQuestion: {ex['question']}\n"
                      f"Answer (say 'unanswerable' if not in context):")
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            out_ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            gen = tokenizer.decode(out_ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            is_impossible = len(ex["answers"]["text"]) == 0
            pred_text = "" if "unanswerable" in gen.lower() else gen
            if metric is not None:
                predictions.append({"id": str(i), "prediction_text": pred_text,
                                     "no_answer_probability": 1.0 if "unanswerable" in gen.lower() else 0.0})
                references.append({"id": str(i),
                                    "answers": ex["answers"] if not is_impossible else {"text": [], "answer_start": []}})
    if metric is not None and predictions:
        result = metric.compute(predictions=predictions, references=references)
        return {"squad_em": result.get("exact", 0.0), "squad_f1": result.get("f1", 0.0)}
    return {"squad_em": float("nan"), "squad_f1": float("nan")}


_GSM8K_ANS_RE = re.compile(r"####\s*(-?[\d,]+)")
_GEN_ANS_RE = re.compile(r"(-?[\d,]+(?:\.\d+)?)")


def eval_gsm8k(model, tokenizer, device, n_samples: int = N_GSM8K, n_shot: int = N_GSM8K_SHOT, max_new_tokens=256) -> Dict[str, float]:
    """
    Grade-school math word problems. Dataset: openai/gsm8k (main config, test split).

    BUG FIX (v8): with a multi-shot prefix prepended, right-truncation
    (the tokenizer default for Llama-3/Qwen-2.5 fast tokenizers) could
    push the target question — or its "Answer:" generation cue — past the
    max_length budget and cut it off, silently corrupting the example
    rather than trimming the (less important) earliest few-shot
    demonstrations. `tokenizer.truncation_side` is now explicitly set to
    "left" for the duration of this eval so truncation always removes the
    front of the prompt first, and is restored to its original value
    afterward so it doesn't leak into other evaluators that may rely on
    default (right) truncation behavior.
    """
    _require_datasets_available()
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n_samples}]")
    fewshot_ds = load_dataset("openai/gsm8k", "main", split=f"train[:{n_shot}]")
    fewshot_prefix = "".join(f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n" for ex in fewshot_ds)

    original_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        correct = total = 0
        with torch.no_grad():
            for ex in ds:
                prompt = f"{fewshot_prefix}Question: {ex['question']}\nAnswer:"
                enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
                out_ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
                gen = tokenizer.decode(out_ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                gold_match = _GSM8K_ANS_RE.search(ex["answer"])
                gold = gold_match.group(1).replace(",", "") if gold_match else None
                pred_match = _GSM8K_ANS_RE.search(gen)
                if pred_match:
                    pred = pred_match.group(1).replace(",", "")
                else:
                    nums = _GEN_ANS_RE.findall(gen)
                    pred = nums[-1].replace(",", "") if nums else None
                if gold is not None and pred is not None:
                    try:
                        correct += int(float(gold) == float(pred))
                    except ValueError:
                        pass
                total += 1
    finally:
        tokenizer.truncation_side = original_truncation_side

    return {"gsm8k_strict_acc": 100.0 * correct / max(total, 1)}


def eval_summarization(model, tokenizer, device, n_samples: int = N_SUMM, max_new_tokens=128) -> Dict[str, float]:
    """Abstractive summarization. Dataset: abisee/cnn_dailymail (config 3.0.0, test split)."""
    _require_datasets_available()
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test[:{n_samples}]")
    rouge = hf_evaluate.load("rouge") if _EVALUATE_AVAILABLE else None
    preds, refs = [], []
    with torch.no_grad():
        for ex in ds:
            prompt = f"Summarize the following article in 2-3 sentences:\n\n{ex['article'][:4000]}\n\nSummary:"
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
            out_ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            gen = tokenizer.decode(out_ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            preds.append(gen)
            refs.append(ex["highlights"])
    if rouge is not None and preds:
        result = rouge.compute(predictions=preds, references=refs)
        return {"rouge1": result["rouge1"] * 100, "rouge2": result["rouge2"] * 100, "rougeL": result["rougeL"] * 100}
    return {"rouge1": float("nan"), "rouge2": float("nan"), "rougeL": float("nan")}


def _load_icd10_synthetic_probe_examples(n_samples: int) -> List[Dict]:
    """
    Builds a synthetic (text -> gold_codes) probe from atta00/icd10-codes,
    which is a code REFERENCE TABLE (chapter/section/category/code/
    description), NOT a labeled clinical-note corpus. Each row's own
    description is wrapped in a templated synthetic "note" and the model is
    asked to recover that row's code. This is a coarse description-recall
    sanity check, explicitly NOT a clinical coding benchmark — there is no
    licensing-clear substitute available (real clinical notes are
    PHI-restricted; MIMIC-III/IV requires a credentialed data-use
    agreement this script has no way to obtain for you).
    """
    _require_datasets_available()
    ds = load_dataset("atta00/icd10-codes", split=f"train[:{n_samples}]")
    examples = []
    for ex in ds:
        desc = (ex.get("description") or "").strip()
        code = (ex.get("code") or ex.get("category_code") or "").strip()
        if not desc or not code:
            continue
        examples.append({
            "text": f"Patient presents with a condition consistent with: {desc.lower()}.",
            "gold_codes": [code],
        })
    if not examples:
        raise CalibrationDataUnavailableError(
            "atta00/icd10-codes loaded but produced zero usable (description, "
            "code) rows — check the dataset's current schema. No fabricated "
            "substitute will be used."
        )
    return examples


def eval_icd10_synthetic_probe(model, tokenizer, device, examples: Optional[List[Dict]] = None,
                                n_samples: int = N_ICD10_SYNTHETIC_PROBE, max_new_tokens=32) -> Dict[str, float]:
    """
    ICD-10 code-description-recall SYNTHETIC PROBE — NOT a clinical coding
    benchmark. See _load_icd10_synthetic_probe_examples docstring. Pass your
    own `examples=[{"text":..., "gold_codes":[...]}]` (e.g. from a licensed
    clinical-note dataset like a MIMIC-derived corpus you have separately
    obtained access to) to get a real clinical-coding result instead.
    """
    if examples is None:
        examples = _load_icd10_synthetic_probe_examples(n_samples)
    tp = fp = fn = top3_correct = 0
    per_code_stats: Dict[str, List[int]] = {}
    with torch.no_grad():
        for ex in examples:
            prompt = f"Clinical note: {ex['text']}\nList the ICD-10 codes (comma-separated):"
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            out_ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            gen = tokenizer.decode(out_ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred_codes = set(c.strip().upper() for c in re.split(r"[,\s]+", gen) if c.strip())
            gold_codes = set(c.upper() for c in ex["gold_codes"])
            tp += len(pred_codes & gold_codes)
            fp += len(pred_codes - gold_codes)
            fn += len(gold_codes - pred_codes)
            if gold_codes & pred_codes:
                top3_correct += 1
            for c in gold_codes | pred_codes:
                stats = per_code_stats.setdefault(c, [0, 0, 0])
                if c in pred_codes and c in gold_codes:
                    stats[0] += 1
                elif c in pred_codes:
                    stats[1] += 1
                elif c in gold_codes:
                    stats[2] += 1
    micro_p = tp / max(tp + fp, 1)
    micro_r = tp / max(tp + fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)
    macro_f1s = []
    for ctp, cfp, cfn in per_code_stats.values():
        p = ctp / max(ctp + cfp, 1)
        r = ctp / max(ctp + cfn, 1)
        macro_f1s.append(2 * p * r / max(p + r, 1e-8))
    macro_f1 = sum(macro_f1s) / max(len(macro_f1s), 1)
    return {
        "icd10_synthetic_probe_micro_f1": micro_f1 * 100,
        "icd10_synthetic_probe_macro_f1": macro_f1 * 100,
        "icd10_synthetic_probe_top3_acc": 100.0 * top3_correct / max(len(examples), 1),
    }


TASK_REGISTRY = {
    "sst2": eval_sst2,
    "squad_v2": eval_squad_v2,
    "gsm8k": eval_gsm8k,
    "summarization": eval_summarization,
    "icd10_synthetic_probe": eval_icd10_synthetic_probe,
}


# ====================================================================
# 7. CALIBRATION TEXT — real data only, loud failure on unavailability
# ====================================================================

def build_task_calibration_texts(task_name: str, n_samples: int) -> List[str]:
    """
    Builds calibration text for ONE task, in that task's own real prompt
    format, from that task's real Hugging Face dataset. Raises
    CalibrationDataUnavailableError (does NOT substitute fabricated
    sentences) if the dataset can't be fetched or yields zero usable rows.
    """
    _require_datasets_available()
    texts: List[str] = []
    try:
        if task_name == "sst2":
            ds = load_dataset("stanfordnlp/sst2", split=f"train[:{n_samples}]")
            texts = [f"Review: {ex['sentence'].strip()}\nSentiment (positive/negative):" for ex in ds]
        elif task_name == "squad_v2":
            ds = load_dataset("rajpurkar/squad_v2", split=f"train[:{n_samples}]")
            texts = [f"Context: {ex['context']}\nQuestion: {ex['question']}\n"
                     f"Answer (say 'unanswerable' if not in context):" for ex in ds]
        elif task_name == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split=f"train[:{n_samples}]")
            texts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in ds]
        elif task_name == "summarization":
            ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"train[:{n_samples}]")
            texts = [f"Summarize the following article in 2-3 sentences:\n\n{ex['article'][:4000]}\n\nSummary:"
                     for ex in ds]
        elif task_name == "icd10_synthetic_probe":
            examples = _load_icd10_synthetic_probe_examples(n_samples)
            texts = [f"Clinical note: {ex['text']}\nList the ICD-10 codes (comma-separated):" for ex in examples]
        else:
            raise KeyError(f"No calibration builder registered for task '{task_name}'")
    except (CalibrationDataUnavailableError, KeyError):
        raise
    except Exception as e:  # noqa: BLE001
        raise CalibrationDataUnavailableError(
            f"Failed to fetch real calibration data for task='{task_name}': {e}. "
            f"This script does not substitute fabricated calibration text — "
            f"fix the underlying network/dataset issue and retry."
        ) from e

    if not texts:
        raise CalibrationDataUnavailableError(
            f"Real calibration dataset for task='{task_name}' loaded but yielded "
            f"zero usable text rows. No fabricated substitute will be used."
        )

    np.random.seed(0)
    np.random.shuffle(texts)
    return texts[:n_samples]


def _load_prior_m0_weights_gb(results_dir: str, model_key: str) -> Optional[float]:
    path = os.path.join(results_dir, f"{model_key}__M0.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("weights_gb")
    except Exception:  # noqa: BLE001
        return None


# ====================================================================
# 8. ORCHESTRATION
# ====================================================================

def run_arm(run_cfg: RunConfig) -> Tuple[dict, dict]:
    spec = resolve_model_spec(run_cfg.model_key)

    results = {"model_key": run_cfg.model_key, "repo_id": spec.repo_id,
               "condition": run_cfg.condition, "timestamp": time.time()}
    audit = {"model_key": run_cfg.model_key, "condition": run_cfg.condition}

    with managed_model(spec, dtype=run_cfg.dtype, device=run_cfg.device) as (model, tokenizer):
        peak_vram_gb(reset=True)

        coverage_report = sanity_check_projection_match(model, run_cfg.model_key)
        audit["projection_coverage"] = coverage_report

        if run_cfg.condition in TASK_SPECIFIC_CONDITIONS:
            result_key = TASK_SPECIFIC_CONDITIONS[run_cfg.condition]
            if run_cfg.condition == "M3":
                task_results, task_audit = run_m3_task_specific(model, tokenizer, run_cfg.device, run_cfg)
            elif run_cfg.condition == "BOSESIR":
                task_results, task_audit = run_bosesir(model, tokenizer, run_cfg.device, run_cfg)
            else:
                raise ValueError(f"Unhandled task-specific condition {run_cfg.condition}")
            results[result_key] = task_results
            audit[result_key] = task_audit
            results["tasks"] = {}
            return results, audit

        if run_cfg.condition == "M0":
            pass
        elif run_cfg.condition == "M1":
            refs = enumerate_linear_projections(model)
            assignment = {name: 8 for name, _, _ in refs}
            per_tensor_audit = replace_with_packed_linear(model, assignment, block_size=run_cfg.bosecomp.quant_block_size)
            audit["per_projection_storage"] = sorted(per_tensor_audit.values(), key=lambda r: (r["layer_idx"], r["proj_type"]))
        elif run_cfg.condition == "M2":
            refs = enumerate_linear_projections(model)
            assignment = {name: 4 for name, _, _ in refs}
            per_tensor_audit = replace_with_packed_linear(model, assignment, block_size=run_cfg.bosecomp.quant_block_size)
            audit["per_projection_storage"] = sorted(per_tensor_audit.values(), key=lambda r: (r["layer_idx"], r["proj_type"]))
        else:
            raise ValueError(f"Unknown condition {run_cfg.condition}")

        results["weights_gb"] = model_weight_footprint_gb(model)
        results["weights_fp16_equivalent_gb"] = model_weight_footprint_fp16_equivalent_gb(model)
        m0_weights_gb = _load_prior_m0_weights_gb(run_cfg.results_dir, run_cfg.model_key)
        results["compression"] = compute_compression_metrics(model, m0_weights_gb)
        results["peak_vram_gb"] = peak_vram_gb()
        results["current_vram_gb_after_load"] = current_vram_gb()

        pre_materialize_model_weights(model)
        task_results = {}
        try:
            for task_name, task_fn in TASK_REGISTRY.items():
                logger.info("Evaluating task: %s", task_name)
                task_results[task_name] = task_fn(model, tokenizer, run_cfg.device)
        finally:
            clear_materialized_model_weights(model)
        results["tasks"] = task_results

    return results, audit


def _existing_result_is_valid(results_dir: str, model_key: str, condition: str) -> bool:
    path = os.path.join(results_dir, f"{model_key}__{condition}.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001
        return False
    if data.get("model_key") != model_key or data.get("condition") != condition:
        return False

    if condition in TASK_SPECIFIC_CONDITIONS:
        result_key = TASK_SPECIFIC_CONDITIONS[condition]
        block = data.get(result_key)
        if not isinstance(block, dict) or not block:
            return False
        return all(t in block and isinstance(block[t], dict) and block[t] for t in TASK_REGISTRY)

    if "weights_gb" not in data:
        return False
    if "tasks" not in data or not isinstance(data["tasks"], dict) or not data["tasks"]:
        return False
    return True


def run_single(model_key: str, condition: str, device: str, dtype: str,
                results_dir: str, force: bool = False, task_n_calib_per_task: int = N_TASK_CALIB_PER_TASK,
                bosesir_n_calib_per_task: Optional[int] = None) -> Optional[str]:
    if not force and _existing_result_is_valid(results_dir, model_key, condition):
        logger.info("RESUME: valid results3/%s__%s.json already exists — skipping (use --force to override).",
                    model_key, condition)
        return os.path.join(results_dir, f"{model_key}__{condition}.json")

    run_cfg = RunConfig(model_key=model_key, condition=condition,
                         device=device, dtype=dtype, results_dir=results_dir,
                         task_n_calib_per_task=task_n_calib_per_task,
                         bosesir_n_calib_per_task=bosesir_n_calib_per_task)
    logger.info("Starting arm: model=%s condition=%s device=%s", model_key, condition, device)
    results, audit = run_arm(run_cfg)

    out_path = run_cfg.result_path()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Wrote results to %s", out_path)

    if len(audit) > 2:
        audit_path = run_cfg.audit_path()
        with open(audit_path, "w", encoding="utf-8") as fh:
            json.dump(audit, fh, indent=2, default=str)
        logger.info("Wrote audit to %s", audit_path)

    return out_path


def run_sweep(models: List[str], device: str, dtype: str, results_dir: str,
              force: bool = False, force_conditions: Optional[List[str]] = None,
              task_n_calib_per_task: int = N_TASK_CALIB_PER_TASK,
              bosesir_n_calib_per_task: Optional[int] = None) -> None:
    ordered_conditions = ["M0"] + [c for c in CONDITIONS if c != "M0"]
    force_set = set(force_conditions or [])
    for model_key in models:
        for condition in ordered_conditions:
            arm_force = force or (condition in force_set)
            # NOTE: unlike the original script, arm failures (including
            # ModelLoadError / CalibrationDataUnavailableError /
            # ProjectionCoverageError / EvidenceSignalDegenerateError) are
            # logged and RE-RAISED to stop the sweep for that model rather
            # than being silently swallowed — a failed arm should be
            # visible, not skipped over.
            run_single(model_key, condition, device, dtype, results_dir, force=arm_force,
                       task_n_calib_per_task=task_n_calib_per_task,
                       bosesir_n_calib_per_task=bosesir_n_calib_per_task)
    aggregate_results(results_dir)


# ====================================================================
# 9. AGGREGATION
# ====================================================================

COLUMNS = [
    "Cond", "SST2_Acc", "SQuAD_F1", "GSM8K_Acc", "Summ_RougeL", "ICD10_SyntheticProbe_MicroF1",
    "Weights_GB", "FP16_Equiv_GB", "Compression_Ratio", "Space_Saved_Pct", "Avg_Bits_Per_Weight",
    "Peak_VRAM_GB",
]

TASK_SPECIFIC_COLUMNS = [
    "Task", "Task_Metric_Primary", "Weights_GB", "FP16_Equiv_GB",
    "Compression_Ratio", "Space_Saved_Pct", "Mean_Thermal_Bits", "N_Pruned_0bit",
    "Calib_N_Samples",
]

_PRIMARY_METRIC_KEY = {
    "sst2": "sst2_acc", "squad_v2": "squad_f1", "gsm8k": "gsm8k_strict_acc",
    "summarization": "rougeL", "icd10_synthetic_probe": "icd10_synthetic_probe_micro_f1",
}


def _load_arm(results_dir: str, model_key: str, condition: str) -> Optional[dict]:
    path = os.path.join(results_dir, f"{model_key}__{condition}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _row_from_result(condition: str, result: Optional[dict]) -> list:
    if result is None:
        return [condition] + ["--"] * (len(COLUMNS) - 1)
    tasks = result.get("tasks", {})
    compression = result.get("compression", {})
    return [
        condition,
        tasks.get("sst2", {}).get("sst2_acc", "--"),
        tasks.get("squad_v2", {}).get("squad_f1", "--"),
        tasks.get("gsm8k", {}).get("gsm8k_strict_acc", "--"),
        tasks.get("summarization", {}).get("rougeL", "--"),
        tasks.get("icd10_synthetic_probe", {}).get("icd10_synthetic_probe_micro_f1", "--"),
        result.get("weights_gb", "--"),
        result.get("weights_fp16_equivalent_gb", "--"),
        compression.get("compression_ratio_vs_fp16_equivalent", "--"),
        compression.get("space_saved_pct_vs_fp16_equivalent", "--"),
        compression.get("effective_avg_bits_per_weight", "--"),
        result.get("peak_vram_gb", "--"),
    ]


def _rows_from_task_specific_result(result: Optional[dict], result_key: str) -> List[list]:
    if result is None or result_key not in result:
        return [[t] + ["--"] * (len(TASK_SPECIFIC_COLUMNS) - 1) for t in TASK_REGISTRY]
    block = result[result_key]
    rows = []
    for task_name in TASK_REGISTRY:
        task_block = block.get(task_name)
        if not task_block:
            rows.append([task_name] + ["--"] * (len(TASK_SPECIFIC_COLUMNS) - 1))
            continue
        metrics = task_block.get("task_metrics", {})
        primary_key = _PRIMARY_METRIC_KEY.get(task_name)
        primary_val = metrics.get(primary_key, "--") if primary_key else "--"
        compression = task_block.get("compression", {})
        bit_summary = task_block.get("bit_assignment_summary", {})
        n_pruned = bit_summary.get("n_pruned_0bit", bit_summary.get("n_pruned_0bit_layers", "--"))
        rows.append([
            task_name, primary_val,
            task_block.get("weights_gb", "--"),
            task_block.get("weights_fp16_equivalent_gb", "--"),
            compression.get("compression_ratio_vs_fp16_equivalent", "--"),
            compression.get("space_saved_pct_vs_fp16_equivalent", "--"),
            bit_summary.get("mean_thermal_bits", "--"),
            n_pruned,
            task_block.get("calib_n_samples", "--"),
        ])
    return rows


def aggregate_results(results_dir: str = RESULTS_DIR) -> None:
    os.makedirs(results_dir, exist_ok=True)
    matrix_conditions = [c for c in CONDITIONS if c not in TASK_SPECIFIC_CONDITIONS]
    for model_key in MODEL_REGISTRY:
        rows = [_row_from_result(cond, _load_arm(results_dir, model_key, cond)) for cond in matrix_conditions]
        if not all(r[1] == "--" for r in rows):
            out_csv = os.path.join(results_dir, f"matrix__{model_key}.csv")
            with open(out_csv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(COLUMNS)
                writer.writerows(rows)
            print(f"Wrote {out_csv}")

        for condition, result_key in TASK_SPECIFIC_CONDITIONS.items():
            task_result = _load_arm(results_dir, model_key, condition)
            if task_result is not None:
                task_rows = _rows_from_task_specific_result(task_result, result_key)
                out_csv_task = os.path.join(results_dir, f"matrix__{model_key}__{result_key}.csv")
                with open(out_csv_task, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(TASK_SPECIFIC_COLUMNS)
                    writer.writerows(task_rows)
                print(f"Wrote {out_csv_task}")


# ====================================================================
# 10. CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Weight-compression benchmark: M0/M1/M2/M3(BoseComp)/BOSESIR. "
                    "Real models only, no fabricated fallback data. "
                    "See kvcache_bench.py for M4/WormComp + other KV-cache methods.")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), help="Run a single model.")
    parser.add_argument("--condition", choices=CONDITIONS, help="Run a single condition.")
    parser.add_argument("--sweep", action="store_true", help="Run all conditions for the given/all models.")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY.keys()),
                         help="Restrict --sweep to these models (default: all).")
    parser.add_argument("--aggregate-only", action="store_true", help="Just rebuild CSV matrices from existing results3/*.json.")
    parser.add_argument("--task-n-calib", type=int, default=N_TASK_CALIB_PER_TASK,
                         help="Calibration sample count PER TASK for M3 and BOSESIR.")
    parser.add_argument("--bosesir-n-calib", type=int, default=None,
                         help="Override calibration sample count PER TASK for BOSESIR specifically.")
    parser.add_argument("--device", default="cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-conditions", nargs="+", choices=CONDITIONS)
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_results(args.results_dir)
        return

    if not _TORCH_AVAILABLE or not _TRANSFORMERS_AVAILABLE:
        logger.error("torch/transformers not available — cannot run model arms.")
        sys.exit(1)

    if args.sweep:
        models = args.models or list(MODEL_REGISTRY.keys())
        run_sweep(models, args.device, args.dtype, args.results_dir,
                  force=args.force, force_conditions=args.force_conditions,
                  task_n_calib_per_task=args.task_n_calib,
                  bosesir_n_calib_per_task=args.bosesir_n_calib)
    elif args.model and args.condition:
        arm_force = args.force or (args.force_conditions is not None and args.condition in args.force_conditions)
        run_single(args.model, args.condition, args.device, args.dtype, args.results_dir,
                   force=arm_force, task_n_calib_per_task=args.task_n_calib,
                   bosesir_n_calib_per_task=args.bosesir_n_calib)
        aggregate_results(args.results_dir)
    else:
        parser.error("Specify --model + --condition, or --sweep, or --aggregate-only.")


if __name__ == "__main__":
    main()
