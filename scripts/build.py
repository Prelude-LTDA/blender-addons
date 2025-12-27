#!/usr/bin/env python3
"""
Build script for Blender addon extensions.

Resolves symlinks to real directories before building, then restores them.
This is necessary because `blender --command extension build` skips symlinks.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Root of the monorepo
ROOT = Path(__file__).parent.parent
ADDONS_DIR = ROOT / "addons"
SHARED_DIR = ROOT / "shared"
DIST_DIR = ROOT / "dist"


def find_blender() -> str:
    """Find the Blender executable."""
    # Check common locations
    candidates = [
        "blender",  # In PATH
        "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS
    ]

    for candidate in candidates:
        if shutil.which(candidate):
            return candidate

    # Check if BLENDER_PATH env var is set
    blender_path = os.environ.get("BLENDER_PATH")
    if blender_path and os.path.isfile(blender_path):
        return blender_path

    raise RuntimeError(
        "Could not find Blender. Set BLENDER_PATH environment variable or add blender to PATH."
    )


def resolve_symlinks(addon_dir: Path) -> list[tuple[Path, Path]]:
    """
    Replace symlinks with actual directory copies.

    Returns list of (symlink_path, target_path) tuples to restore later.
    """
    resolved: list[tuple[Path, Path]] = []

    for item in addon_dir.iterdir():
        if item.is_symlink():
            target = item.resolve()
            print(f"  Resolving symlink: {item.name} -> {target}")

            # Store info for restoration
            resolved.append((item, target))

            # Remove symlink and copy actual directory
            item.unlink()
            if target.is_dir():
                shutil.copytree(target, item)
            else:
                shutil.copy2(target, item)

    return resolved


def restore_symlinks(resolved: list[tuple[Path, Path]]) -> None:
    """Restore symlinks after build."""
    for symlink_path, target_path in resolved:
        print(f"  Restoring symlink: {symlink_path.name}")

        # Remove copied directory/file
        if symlink_path.is_dir():
            shutil.rmtree(symlink_path)
        else:
            symlink_path.unlink()

        # Recreate symlink (relative path)
        rel_target = os.path.relpath(target_path, symlink_path.parent)
        symlink_path.symlink_to(rel_target)


def build_addon(addon_name: str, blender: str) -> Path | None:
    """Build a single addon extension."""
    addon_dir = ADDONS_DIR / addon_name

    if not addon_dir.exists():
        print(f"Error: Addon directory not found: {addon_dir}")
        return None

    if not (addon_dir / "blender_manifest.toml").exists():
        print(f"Error: No blender_manifest.toml in {addon_dir}")
        return None

    print(f"\nBuilding {addon_name}...")

    # Resolve symlinks
    resolved = resolve_symlinks(addon_dir)

    try:
        # Create dist directory
        DIST_DIR.mkdir(exist_ok=True)

        # Build the extension
        result = subprocess.run(
            [
                blender,
                "--command",
                "extension",
                "build",
                "--source-dir",
                str(addon_dir),
                "--output-dir",
                str(DIST_DIR),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Build failed for {addon_name}:")
            print(result.stderr)
            return None

        print(result.stdout)

        # Find the built zip file
        for f in DIST_DIR.iterdir():
            if f.name.startswith(addon_name) and f.suffix == ".zip":
                print(f"  Built: {f}")
                return f

    finally:
        # Always restore symlinks
        restore_symlinks(resolved)

    return None


def build_all(blender: str) -> list[Path]:
    """Build all addons in the addons directory."""
    built: list[Path] = []

    for addon_dir in ADDONS_DIR.iterdir():
        if addon_dir.is_dir() and (addon_dir / "blender_manifest.toml").exists():
            result = build_addon(addon_dir.name, blender)
            if result:
                built.append(result)

    return built


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Blender addon extensions")
    parser.add_argument(
        "addon",
        nargs="?",
        help="Name of specific addon to build (builds all if not specified)",
    )
    parser.add_argument(
        "--blender",
        help="Path to Blender executable",
    )

    args = parser.parse_args()

    try:
        blender = args.blender or find_blender()
        print(f"Using Blender: {blender}")
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    if args.addon:
        result = build_addon(args.addon, blender)
        return 0 if result else 1
    else:
        results = build_all(blender)
        print(f"\nBuilt {len(results)} addon(s)")
        return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
