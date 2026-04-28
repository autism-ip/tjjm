"""
 * [INPUT]: 依赖 pathlib, pytest.monkeypatch, src.models.weights 的缓存与加载逻辑
 * [OUTPUT]: 对外提供 weights 模块的单元测试
 * [POS]: tests/unit/ 的权重缓存验证器，覆盖项目内缓存路径收口与下载入口参数
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from pathlib import Path
import warnings

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
    assert Path(calls["filepath"]) == expected
    assert Path(calls["loaded_path"]) == expected
