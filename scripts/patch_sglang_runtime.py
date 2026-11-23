#!/usr/bin/env python3
"""Apply narrowly scoped fixes to the pinned SGLang inference runtime."""

from __future__ import annotations

from pathlib import Path


ORIGINAL_FREE_PORT = """def get_free_port():
    sock = try_bind_socket()
"""
PATCHED_FREE_PORT = """def get_free_port():
    # Do not select a TIME_WAIT port that Torch TCPStore cannot bind after the
    # SGLang data-parallel controller releases its reservation socket.
    sock = try_bind_socket(reuse_addr=False)
"""


def patch_sglang_network(network_path: Path) -> bool:
    """Patch SGLang's port probe; return whether the file changed."""

    source = network_path.read_text(encoding="utf-8")
    if PATCHED_FREE_PORT in source:
        return False
    if ORIGINAL_FREE_PORT not in source:
        raise RuntimeError(
            "the pinned SGLang network implementation changed; refusing to "
            f"patch an unknown file: {network_path}"
        )
    network_path.write_text(
        source.replace(ORIGINAL_FREE_PORT, PATCHED_FREE_PORT, 1),
        encoding="utf-8",
    )
    return True


def main() -> None:
    from sglang.srt.utils import network

    network_path = Path(network.__file__).resolve()
    changed = patch_sglang_network(network_path)
    state = "patched" if changed else "already patched"
    print(f"SGLang TCPStore port selection: {state} ({network_path})")


if __name__ == "__main__":
    main()
