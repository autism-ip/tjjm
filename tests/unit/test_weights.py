"""
 * [INPUT]: 依赖 pathlib, pytest.monkeypatch, src.models.weights 的缓存与加载逻辑
 * [OUTPUT]: 对外提供 weights 模块的单元测试
 * [POS]: tests/unit/ 的权重缓存验证器，覆盖项目内缓存路径收口与下载入口参数
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from pathlib import Path
import warnings

import pytest

from src.models import weights


def test_resolve_pretrained_cache_path_defaults_to_project_cache(monkeypatch):
    """未设置 XDG_CACHE_HOME 时，应回退到项目内 .cache。"""
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    path = weights._resolve_pretrained_cache_path()

    expected = (
        Path(weights.__file__).resolve().parents[2]
        / ".cache"
        / "monai"
        / "pretrained"
        / "ssl_pretrained_weights.pth"
    )
    assert path == expected
    assert path.parent.exists()


def test_load_pretrained_uses_project_cache_path(monkeypatch, tmp_path):
    """自动下载时应写入可写缓存目录，而不是用户家目录。"""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache-root"))

    calls: dict[str, str] = {}

    def fake_download(url: str, filepath: str) -> None:
        calls["url"] = url
        calls["filepath"] = filepath
        Path(filepath).write_bytes(b"fake")

    def fake_load(model, path: str) -> None:
        calls["loaded_path"] = path

    monkeypatch.setattr("monai.apps.download_url", fake_download)
    monkeypatch.setattr(weights, "_load_weights_file", fake_load)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        weights.load_swin_unetr_pretrained(model=object(), checkpoint_path=None)

    expected = tmp_path / "cache-root" / "monai" / "pretrained" / "ssl_pretrained_weights.pth"
    assert calls["url"] == weights.SSL_PRETRAINED_URL
    assert Path(calls["filepath"]) == expected.with_suffix(".pth.part")
    assert Path(calls["loaded_path"]) == expected


def test_load_pretrained_prefers_explicit_local_checkpoint(monkeypatch, tmp_path):
    """Explicit local weights must bypass network download for reproducible runs."""
    local_checkpoint = tmp_path / "ssl_pretrained_weights.pth"
    local_checkpoint.write_bytes(b"fake")

    calls: dict[str, str] = {}

    def fake_load(model, path: str):
        calls["loaded_path"] = path
        return weights.PretrainedLoadResult(
            status="loaded",
            source=Path(path),
            loaded_layers=3,
            not_loaded_layers=1,
            message="loaded",
        )

    def fail_download(_local_path: Path):
        raise AssertionError("explicit checkpoint should not download")

    monkeypatch.setattr(weights, "_load_weights_file", fake_load)
    monkeypatch.setattr(weights, "_download_pretrained_weights", fail_download)

    result = weights.load_swin_unetr_pretrained(
        model=object(),
        checkpoint_path=str(local_checkpoint),
        strict=True,
    )

    assert result.status == "loaded"
    assert result.loaded_layers == 3
    assert Path(calls["loaded_path"]) == local_checkpoint


def test_load_pretrained_strict_mode_raises_on_failure(monkeypatch, tmp_path):
    """Strict mode must fail fast instead of silently using random initialization."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache-root"))

    def fail_download(_local_path: Path):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(weights, "_download_pretrained_weights", fail_download)

    with pytest.raises(RuntimeError, match="network unavailable"):
        weights.load_swin_unetr_pretrained(
            model=object(),
            checkpoint_path=None,
            strict=True,
        )


def test_load_pretrained_retries_after_corrupt_cached_file(monkeypatch, tmp_path):
    """Corrupt cached weights should be deleted and downloaded again once."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache-root"))

    download_calls: list[Path] = []
    load_calls: list[str] = []
    cache_path = weights._resolve_pretrained_cache_path()

    def fake_download(local_path: Path) -> Path:
        download_calls.append(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"new-file")
        return local_path

    def fake_load(model, path: str):
        load_calls.append(path)
        if len(load_calls) == 1:
            raise RuntimeError("corrupt archive")
        return weights.PretrainedLoadResult(
            status="loaded",
            source=Path(path),
            loaded_layers=5,
            not_loaded_layers=2,
            message="loaded",
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"bad-file")
    monkeypatch.setattr(weights, "_download_pretrained_weights", fake_download)
    monkeypatch.setattr(weights, "_load_weights_file", fake_load)

    result = weights.load_swin_unetr_pretrained(model=object(), checkpoint_path=None)

    assert result.status == "loaded"
    assert len(download_calls) == 2
    assert len(load_calls) == 2
    assert cache_path.exists()
