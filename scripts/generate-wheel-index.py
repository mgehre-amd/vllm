#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generate a PyPI-compatible simple repository index for wheel files.

This script creates HTML index files that can be served via GitHub Pages
or any static file server, allowing pip to install wheels using --extra-index-url.

Usage:
    python generate-wheel-index.py --wheels-dir dist/ --output-dir gh-pages/wheels

The generated structure:
    gh-pages/wheels/
        index.html          # Top-level index listing packages
        vllm/
            index.html      # Package index listing wheel files
        *.whl               # Wheel files (copied or linked)
"""

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


@dataclass
class WheelInfo:
    """Parsed wheel file metadata."""

    package_name: str
    version: str
    python_tag: str
    abi_tag: str
    platform_tag: str
    variant: str | None
    filename: str


def parse_wheel_filename(filename: str) -> WheelInfo:
    """
    Parse wheel filename to extract metadata.

    Format: {package}-{version}(-{build})?-{python}-{abi}-{platform}.whl

    Examples:
        vllm-0.11.0-cp38-abi3-manylinux_2_31_x86_64.whl
        vllm-0.11.0+rocm6.3-cp312-cp312-manylinux_2_31_x86_64.whl
    """
    # Remove .whl extension
    base = filename.removesuffix(".whl")

    # Split from the right to handle versions with dashes
    parts = base.rsplit("-", 4)
    if len(parts) < 5:
        raise ValueError(f"Invalid wheel filename: {filename}")

    package_name = parts[0]
    version = parts[1]
    python_tag = parts[2]
    abi_tag = parts[3]
    platform_tag = parts[4]

    # Extract variant from version (e.g., +rocm6.3, +cu129)
    variant = None
    if "+" in version:
        version_base, variant = version.split("+", 1)

    return WheelInfo(
        package_name=package_name,
        version=version,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
        variant=variant,
        filename=filename,
    )


def generate_top_index(packages: list[str], base_url: str) -> str:
    """Generate top-level index.html listing all packages."""
    timestamp = datetime.now().isoformat()

    package_links = "\n".join(
        f'    <a href="{pkg}/">{pkg}</a><br/>' for pkg in sorted(packages)
    )

    # noqa: E501 - HTML templates have long lines
    return f"""<!DOCTYPE html>
<html>
  <head>
    <meta name="pypi:repository-version" content="1.0">
    <title>vLLM ROCm Wheels</title>
    <style>
      body {{ font-family: sans-serif; max-width: 800px; margin: 40px auto; }}
      code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
      pre {{ background: #f4f4f4; padding: 12px; border-radius: 6px; }}
      a {{ color: #0366d6; }}
    </style>
  </head>
  <body>
    <h1>🚀 vLLM ROCm Wheels</h1>
    <p><em>Generated: {timestamp}</em></p>
    
    <h2>Installation</h2>
    <pre>pip install vllm --extra-index-url {base_url}</pre>
    
    <p><strong>Note:</strong> Install ROCm PyTorch first:</p>
    <pre>pip install torch --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1151</pre>
    
    <h2>Available Packages</h2>
{package_links}
  </body>
</html>
"""


def generate_package_index(
    wheels: list[WheelInfo], wheel_dir_relative: str = ".."
) -> str:
    """Generate package-level index.html listing wheel files."""
    wheel_links = []

    for wheel in sorted(wheels, key=lambda w: w.version, reverse=True):
        # URL-encode the filename (especially for + signs)
        encoded_filename = quote(wheel.filename, safe="")
        href = f"{wheel_dir_relative}/{encoded_filename}"
        wheel_links.append(f'    <a href="{href}">{wheel.filename}</a><br/>')

    return f"""<!DOCTYPE html>
<html>
  <head>
    <meta name="pypi:repository-version" content="1.0">
    <title>vllm</title>
  </head>
  <body>
{chr(10).join(wheel_links)}
  </body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyPI-compatible wheel index for GitHub Pages"
    )
    parser.add_argument(
        "--wheels-dir",
        type=Path,
        required=True,
        help="Directory containing wheel files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the index (e.g., gh-pages/wheels)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://example.github.io/vllm/wheels",
        help="Base URL where the index will be hosted",
    )
    parser.add_argument(
        "--copy-wheels",
        action="store_true",
        help="Copy wheel files to output directory (default: just generate index)",
    )

    args = parser.parse_args()

    wheels_dir: Path = args.wheels_dir
    output_dir: Path = args.output_dir

    # Find all wheel files
    wheel_files = list(wheels_dir.glob("*.whl"))
    if not wheel_files:
        print(f"No wheel files found in {wheels_dir}")
        return

    print(f"Found {len(wheel_files)} wheel files")

    # Parse wheel metadata
    wheels_by_package: dict[str, list[WheelInfo]] = {}
    for whl_path in wheel_files:
        try:
            info = parse_wheel_filename(whl_path.name)
            if info.package_name not in wheels_by_package:
                wheels_by_package[info.package_name] = []
            wheels_by_package[info.package_name].append(info)
            print(f"  - {info.filename} ({info.package_name} {info.version})")
        except ValueError as e:
            print(f"  - Skipping {whl_path.name}: {e}")

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy wheels if requested
    if args.copy_wheels:
        for whl_path in wheel_files:
            dest = output_dir / whl_path.name
            shutil.copy2(whl_path, dest)
            print(f"Copied {whl_path.name}")

    # Generate top-level index
    top_index = generate_top_index(list(wheels_by_package.keys()), args.base_url)
    (output_dir / "index.html").write_text(top_index)
    print(f"Generated {output_dir / 'index.html'}")

    # Generate package indices
    for package_name, wheels in wheels_by_package.items():
        package_dir = output_dir / package_name
        package_dir.mkdir(exist_ok=True)

        package_index = generate_package_index(wheels)
        (package_dir / "index.html").write_text(package_index)
        print(f"Generated {package_dir / 'index.html'} with {len(wheels)} wheels")

    print("\n✅ Index generated successfully!")
    print("\nTo use this index:")
    print(f"  pip install vllm --extra-index-url {args.base_url}")


if __name__ == "__main__":
    main()
