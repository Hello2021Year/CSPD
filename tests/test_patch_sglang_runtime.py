from pathlib import Path

import pytest

from scripts.patch_sglang_runtime import (
    PATCHED_FREE_PORT,
    patch_sglang_network,
)


def test_patch_sglang_network_is_narrow_and_idempotent(tmp_path: Path) -> None:
    network_path = tmp_path / "network.py"
    network_path.write_text(
        "before\ndef get_free_port():\n    sock = try_bind_socket()\n"
        "    return sock\nafter\n",
        encoding="utf-8",
    )

    assert patch_sglang_network(network_path) is True
    patched = network_path.read_text(encoding="utf-8")
    assert PATCHED_FREE_PORT in patched
    assert patched.startswith("before\n")
    assert patched.endswith("after\n")
    assert patch_sglang_network(network_path) is False


def test_patch_sglang_network_rejects_unknown_source(tmp_path: Path) -> None:
    network_path = tmp_path / "network.py"
    network_path.write_text("def get_free_port(): pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to patch an unknown file"):
        patch_sglang_network(network_path)
