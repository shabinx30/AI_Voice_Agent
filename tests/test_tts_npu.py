"""Tests for the static NPU Kokoro encoder helpers (no hardware needed)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from app.config import Settings
from app.core.tts_npu import (
    NPU_MAX_TOKENS,
    DurationEncoderStatic,
    KokoroNPUEncoder,
    ProsodyBodyStatic,
    TextEncoderStatic,
    pad_inputs,
)


def test_pad_inputs_shapes_and_masks() -> None:
    """Verifies padding, attention mask, and pad mask semantics."""
    ids = np.array([[0, 53, 83, 4, 0]], dtype=np.int64)
    padded, attn, tmask, n = pad_inputs(ids, 217)
    assert n == 5
    assert padded.shape == (1, 217)
    assert list(padded[0, :5]) == [0, 53, 83, 4, 0]
    assert (padded[0, 5:] == 0).all()
    assert attn.shape == (1, 217) and attn[0, :5].all() and not attn[0, 5:].any()
    assert tmask.shape == (1, 217) and not tmask[0, :5].any() and tmask[0, 5:].all()


def test_pad_inputs_overflow_raises() -> None:
    """Verifies over-length token sequences are rejected for the NPU path."""
    with pytest.raises(ValueError):
        pad_inputs(np.zeros((1, NPU_MAX_TOKENS + 1), dtype=np.int64))


def test_npu_stage_placement_defaults() -> None:
    """Verifies albert is CPU-pinned by default but forcible onto NPU."""
    enc = KokoroNPUEncoder()
    assert enc._placement("albert", "NPU") == ["CPU"]
    assert enc._placement("prosody_body", "NPU") == ["NPU", "CPU"]
    assert enc._placement("text_encoder", "NPU") == ["NPU", "CPU"]
    full = KokoroNPUEncoder(cpu_stages=frozenset())
    assert full._placement("albert", "NPU") == ["NPU", "CPU"]
    assert not enc.is_loaded
    assert enc.npu_stage_count == 0


def test_text_encoder_static_shapes() -> None:
    """Verifies the pack-free TextEncoder replica runs with tiny dims."""
    src = SimpleNamespace(
        embedding=nn.Embedding(10, 8),
        cnn=nn.ModuleList([]),
        lstm=nn.LSTM(8, 4, 1, batch_first=True, bidirectional=True),
    )
    rep = TextEncoderStatic(src).eval()
    with torch.no_grad():
        out = rep(torch.randint(0, 10, (1, 5)), torch.zeros(1, 5, dtype=torch.bool))
    assert out.shape == (1, 8, 5)


def test_prosody_body_static_shapes() -> None:
    """Verifies the pack-free prosody replica outputs shapes and int durations."""
    dur_src = SimpleNamespace(
        lstms=nn.ModuleList([nn.LSTM(10, 3, 1, batch_first=True, bidirectional=True)]),
        dropout=0.0,
    )
    predictor = SimpleNamespace(
        text_encoder=dur_src,
        lstm=nn.LSTM(6, 3, 1, batch_first=True, bidirectional=True),
        duration_proj=nn.Linear(6, 7),
    )
    rep = ProsodyBodyStatic(predictor).eval()
    with torch.no_grad():
        d, pred_dur = rep(
            torch.randn(1, 6, 5),
            torch.randn(1, 4),
            torch.zeros(1, 5, dtype=torch.bool),
        )
    assert d.shape == (1, 5, 6)
    assert pred_dur.shape == (1, 5)
    assert pred_dur.dtype == torch.int64
    assert bool((pred_dur >= 1).all())


def test_tts_npu_full_defaults_off() -> None:
    """Verifies the full-NPU fidelity tradeoff stays opt-in by default."""
    assert Settings().tts_npu_full is False
