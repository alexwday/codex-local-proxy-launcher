#!/usr/bin/env python3
"""Restore Codex config changes applied by the local proxy launcher."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from codex_config_manager import restart_codex_desktop, restore_codex_config
from config import Config

SRC_DIR = Path(__file__).parent.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="Restart Codex Desktop after restoring config")
    args = parser.parse_args()

    load_dotenv(SRC_DIR / ".env")
    config = Config()
    status = restore_codex_config(config)
    print(status.get("message", "Restore attempted."))
    print(f"Config: {status.get('config_path')}")
    print(f"Applied: {status.get('is_applied')}")
    if args.restart:
        restart_status = restart_codex_desktop(config)
        print(f"Codex running: {restart_status.get('codex_running')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
