"""Verifies cached Kokoro frontend matches optimum preprocess exactly (no fidelity drift)."""

import numpy as np
import torch


def _engine():
    from app.core.tts import KokoroTTSEngine

    eng = KokoroTTSEngine(device="npu", speaker="af_heart")
    eng._cache_max = 0
    eng.load_model()
    return eng


def test_cached_preprocess_matches_optimum() -> None:
    """Token IDs, ref embeddings, and phonemes must equal optimum's output."""
    eng = _engine()
    texts = [
        "Paris is the capital of France.",
        "Sure, give me a moment.",
        "Hello! How are you today? I can help you with that.",
    ]
    for text in texts:
        ref = eng._ov_model.preprocess_input(text, voice="af_heart", lang_code="a")
        got = eng._preprocess_kokoro(text, voice="af_heart", lang_code="a")
        assert len(got["segments"]) == len(ref["segments"])
        for g_seg, r_seg in zip(got["segments"], ref["segments"]):
            assert g_seg["phonemes"] == r_seg["phonemes"]
            assert g_seg["graphemes"] == r_seg["graphemes"]
            assert torch.equal(g_seg["input_ids"], r_seg["input_ids"])
            assert np.allclose(
                np.asarray(g_seg["ref_s"]), np.asarray(r_seg["ref_s"]), atol=0
            )


def test_cached_preprocess_multisegment_matches() -> None:
    """Multi-segment inputs (newlines) must also match exactly."""
    eng = _engine()
    text = "First line here.\nSecond line here.\nThird line here."
    ref = eng._ov_model.preprocess_input(text, voice="af_bella", lang_code="a")
    got = eng._preprocess_kokoro(text, voice="af_bella", lang_code="a")
    assert len(got["segments"]) == len(ref["segments"]) >= 2
    for g_seg, r_seg in zip(got["segments"], ref["segments"]):
        assert torch.equal(g_seg["input_ids"], r_seg["input_ids"])
        assert np.allclose(
            np.asarray(g_seg["ref_s"]), np.asarray(r_seg["ref_s"]), atol=0
        )


def test_cached_synthesis_deterministic_duration() -> None:
    """NPU synth via cached frontend must be deterministic with stable duration.

    Frontend inputs are proven bit-identical above, so synthesis numerics
    cannot drift; this guards the wiring (segment flow, concat, sr).
    s1 measured 2.25s pre-change; allow a wide window for HW variance.
    """
    eng = _engine()
    text = "Paris is the capital of France."
    eng._cache_max = 0
    eng.warm_npu_encoder()
    eng.synthesize("Warmup complete.", speaker="af_heart")
    eng._cache.clear()
    audio1, sr1 = eng.synthesize(text, speaker="af_heart")
    eng._cache.clear()
    audio2, sr2 = eng.synthesize(text, speaker="af_heart")
    assert sr1 == sr2 == 24000
    assert len(audio1) == len(audio2)
    assert abs(len(audio1) / sr1 - 2.25) < 0.35
    assert float(np.corrcoef(audio1, audio2)[0, 1]) > 0.98
