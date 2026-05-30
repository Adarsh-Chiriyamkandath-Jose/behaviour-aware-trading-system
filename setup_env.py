"""One-command environment setup for the Behaviour-Aware Trading System.

Creates a local virtual environment (.venv), upgrades pip, and installs
everything in requirements.txt. Works the same on Windows, macOS and Linux, so
nobody has to install packages by hand.

    python setup_env.py

Then activate the environment when prompted at the end.
"""
from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

VENV_DIR = Path(".venv")
REQUIREMENTS = Path("requirements.txt")


def venv_python(venv_dir: Path) -> Path:
    """Path to the python executable inside the virtual environment."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    if not REQUIREMENTS.exists():
        sys.exit(f"Could not find {REQUIREMENTS}. Run this from the project root.")

    if not VENV_DIR.exists():
        print(f"Creating virtual environment in {VENV_DIR} ...")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print(f"Reusing existing virtual environment in {VENV_DIR}")

    py = venv_python(VENV_DIR)
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    activate = (
        ".venv\\Scripts\\activate"
        if os.name == "nt"
        else "source .venv/bin/activate"
    )
    print("\nDone. Activate the environment with:\n")
    print(f"    {activate}\n")


if __name__ == "__main__":
    main()
