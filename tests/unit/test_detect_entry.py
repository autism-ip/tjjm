"""
 * [INPUT]: 依赖 numpy, omegaconf, scripts.detect
 * [OUTPUT]: 对外提供 detect 入口契约测试
 * [POS]: tests/unit/ 的检测入口验证器，覆盖评估数组规整与旧 checkpoint 的编码器回退行为
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import numpy as np
from omegaconf import OmegaConf
from omegaconf import ListConfig


def test_detect_evaluation_arrays_accept_numpy_outputs():
    from scripts.detect import _collect_evaluation_arrays

    results = {
        "case_a": {
            "anomaly_map": np.array([[0.1, 0.9]], dtype=np.float32),
            "ground_truth": np.array([[0, 1]], dtype=np.uint8),
        },
        "case_b": {
            "anomaly_map": np.array([[0.2, 0.8]], dtype=np.float32),
            "ground_truth": np.array([[0, 1]], dtype=np.uint8),
        },
    }

    preds, gts = _collect_evaluation_arrays(results)

    np.testing.assert_allclose(preds, np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32))
    np.testing.assert_array_equal(gts, np.array([0, 1, 0, 1], dtype=np.uint8))


def test_detect_checkpoint_without_encoder_name_uses_supported_default(monkeypatch, tmp_path):
    from scripts import detect

    captured = {}

    class DummyModel:
        def __init__(self, encoder_name, pretrained):
            captured["encoder_name"] = encoder_name
            captured["pretrained"] = pretrained

        def load_state_dict(self, state_dict):
            captured["state_dict"] = state_dict

        def to(self, device):
            captured["device"] = str(device)
            return self

        def eval(self):
            captured["eval_called"] = True
            return self

    class DummyDetector:
        def __init__(self, model, patch_size, stride, batch_size, device):
            captured["detector_args"] = {
                "patch_size": patch_size,
                "stride": stride,
                "batch_size": batch_size,
                "device": str(device),
            }

        def run_directory(self, test_ct_dir):
            captured["test_ct_dir"] = test_ct_dir
            return {}

    monkeypatch.setattr(detect, "Autoencoder3D", DummyModel)
    monkeypatch.setattr(detect, "SlidingWindowDetector", DummyDetector)
    monkeypatch.setattr(detect, "setup_logging", lambda: None)
    monkeypatch.setattr(
        detect.torch,
        "load",
        lambda path, map_location: {"state_dict": {"model.weight": 1}},
    )

    cfg = OmegaConf.create(
        {
            "data": {
                "output_dir": str(tmp_path / "outputs"),
                "test_ct_dir": str(tmp_path / "ct"),
            },
            "model": {
                "checkpoint_path": str(tmp_path / "model.ckpt"),
            },
            "detection": {
                "patch_size": [64, 64, 64],
                "stride": 32,
                "batch_size": 2,
            },
        }
    )

    detect.main.__wrapped__(cfg)

    assert captured["encoder_name"] == "swin_unetr"
    assert captured["pretrained"] is False
    assert captured["state_dict"] == {"weight": 1}
    assert captured["eval_called"] is True


def test_detect_strips_lightning_model_prefix_from_state_dict():
    from scripts.detect import _normalize_state_dict_keys

    assert _normalize_state_dict_keys({"model.weight": 1, "bias": 2}) == {"weight": 1, "bias": 2}
    assert _normalize_state_dict_keys({"weight": 1}) == {"weight": 1}


def test_detect_resolves_stride_from_scalar_or_sequence():
    from scripts.detect import _resolve_stride

    assert _resolve_stride(32) == 32
    assert _resolve_stride([32, 32, 32]) == 32
    assert _resolve_stride((16, 16, 16)) == 16
    assert _resolve_stride(ListConfig([8, 8, 8])) == 8
