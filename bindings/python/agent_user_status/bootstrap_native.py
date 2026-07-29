"""Native monitor installation helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from agent_user_status.bootstrap_support import (
    BootstrapPaths,
    native_app_bundle,
    native_app_executable,
)


class Logger(Protocol):
    def __call__(self, level: str, message: str) -> None: ...


def install_native_monitor(paths: BootstrapPaths, log: Logger) -> None:
    native_bin = paths.bin_dir / "agent-user-status-native-monitor"
    native_source_dir = paths.share_dir / "native-monitor"
    native_app = native_app_bundle(paths)
    native_app_macos = native_app / "Contents" / "MacOS"
    native_app_resources = native_app / "Contents" / "Resources"
    native_source_dir.mkdir(parents=True, exist_ok=True)
    native_app_macos.mkdir(parents=True, exist_ok=True)
    native_app_resources.mkdir(parents=True, exist_ok=True)

    native_sources = sorted((paths.src / "native" / "macos").glob("*.swift"))
    for src in native_sources:
        shutil.copy2(src, native_source_dir / src.name)
    shutil.copy2(paths.root / "packaging" / "macos" / "Info.plist", native_app / "Contents" / "Info.plist")

    if sys.platform != "darwin":
        log("info", "non-macOS platform detected; skipping swift compile.")
        return
    if not shutil.which("swiftc"):
        log("warn", "swiftc missing. Install Xcode Command Line Tools to build tray monitor.")
        return

    subprocess.run(_swift_compile_command(native_sources, native_app_executable(paths)), check=True)
    native_app_executable(paths).chmod(0o700)
    shutil.copy2(native_app_executable(paths), native_bin)
    native_bin.chmod(0o700)
    log("info", f"compiled tray monitor app to {native_app}")


def _swift_compile_command(native_sources: list[Path], output: Path) -> list[str]:
    return [
        "swiftc",
        *[str(path) for path in native_sources],
        "-o",
        str(output),
        "-framework",
        "AppKit",
        "-framework",
        "CoreGraphics",
    ]
