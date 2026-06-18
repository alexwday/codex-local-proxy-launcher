"""Manage the user-level Codex config override for this local proxy."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import tomlkit

STATE_FILENAME = "codex_config_state.json"
BACKUP_DIRNAME = "backups"


def _read_toml(path: Path):
    if path.exists():
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    return tomlkit.document()


def _write_toml(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(document), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _state_path(config) -> Path:
    return Path(config.launcher_state_dir) / STATE_FILENAME


def _backup_dir(config) -> Path:
    return Path(config.launcher_state_dir) / BACKUP_DIRNAME


def _provider_table_toml(provider_id: str, provider_table: Any) -> str:
    doc = tomlkit.document()
    providers = tomlkit.table()
    providers.add(provider_id, copy.deepcopy(provider_table))
    doc.add("model_providers", providers)
    return tomlkit.dumps(doc)


def _load_provider_table(provider_id: str, provider_toml: str):
    doc = tomlkit.parse(provider_toml)
    return doc.get("model_providers", {}).get(provider_id)


def ensure_proxy_token_file(config) -> None:
    """Persist the current proxy token where Codex command auth can read it."""
    token_file = Path(config.proxy_token_file)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    current = token_file.read_text(encoding="utf-8").strip() if token_file.exists() else ""
    if current != config.proxy_access_token:
        token_file.write_text(config.proxy_access_token + "\n", encoding="utf-8")
    try:
        token_file.chmod(0o600)
    except OSError:
        pass


def capture_original_state(config, doc) -> dict[str, Any]:
    """Capture only the settings this launcher changes."""
    provider_id = config.codex_provider_id
    providers = doc.get("model_providers", {})
    had_provider = provider_id in providers
    provider_table = providers.get(provider_id) if had_provider else None
    return {
        "created_at": int(time.time()),
        "config_path": str(config.codex_config_path),
        "provider_id": provider_id,
        "had_model": "model" in doc,
        "model": doc.get("model"),
        "had_model_provider": "model_provider" in doc,
        "model_provider": doc.get("model_provider"),
        "had_provider": had_provider,
        "provider_toml": _provider_table_toml(provider_id, provider_table) if had_provider else "",
    }


def load_state(config) -> dict[str, Any] | None:
    path = _state_path(config)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_state(config, state: dict[str, Any]) -> None:
    path = _state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def write_backup(config) -> str | None:
    config_path = Path(config.codex_config_path)
    if not config_path.exists():
        return None
    backup_dir = _backup_dir(config)
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"config.{timestamp}.toml"
    backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        backup_path.chmod(0o600)
    except OSError:
        pass
    return str(backup_path)


def apply_codex_config(config) -> dict[str, Any]:
    """Apply the managed Codex provider override to user-level config.toml."""
    ensure_proxy_token_file(config)
    config_path = Path(config.codex_config_path)
    doc = _read_toml(config_path)
    state = load_state(config)
    backup_path = write_backup(config)

    if state is None:
        state = capture_original_state(config, doc)
    state["last_applied_at"] = int(time.time())
    state["last_backup_path"] = backup_path
    write_state(config, state)

    doc["model"] = config.default_model or "gpt-5.5"
    doc["model_provider"] = config.codex_provider_id

    providers = doc.get("model_providers")
    if providers is None:
        providers = tomlkit.table()
        doc["model_providers"] = providers

    provider = tomlkit.table()
    provider["name"] = config.codex_provider_name
    provider["base_url"] = config.get_openai_base_url()
    provider["wire_api"] = "responses"
    provider["supports_websockets"] = False
    provider["stream_idle_timeout_ms"] = config.streaming_timeout_seconds * 1000
    auth = tomlkit.table()
    auth["command"] = "/bin/cat"
    auth["args"] = [str(config.proxy_token_file)]
    auth["refresh_interval_ms"] = 0
    provider["auth"] = auth
    providers[config.codex_provider_id] = provider

    _write_toml(config_path, doc)
    return get_codex_status(config)


def restore_codex_config(config) -> dict[str, Any]:
    """Restore the Codex settings captured before the first apply."""
    state = load_state(config)
    if state is None:
        return get_codex_status(config) | {"restored": False, "message": "No saved Codex config state found."}

    config_path = Path(config.codex_config_path)
    doc = _read_toml(config_path)
    provider_id = state.get("provider_id") or config.codex_provider_id

    if state.get("had_model"):
        doc["model"] = state.get("model")
    else:
        doc.pop("model", None)

    if state.get("had_model_provider"):
        doc["model_provider"] = state.get("model_provider")
    else:
        doc.pop("model_provider", None)

    providers = doc.get("model_providers")
    if providers is None and state.get("had_provider"):
        providers = tomlkit.table()
        doc["model_providers"] = providers

    if providers is not None:
        if state.get("had_provider"):
            original_provider = _load_provider_table(provider_id, state.get("provider_toml", ""))
            if original_provider is not None:
                providers[provider_id] = original_provider
        else:
            providers.pop(provider_id, None)

    _write_toml(config_path, doc)
    try:
        _state_path(config).unlink()
    except FileNotFoundError:
        pass
    return get_codex_status(config) | {"restored": True, "message": "Codex config restored."}


def _codex_process_lines(config) -> list[str]:
    patterns = [
        str(Path(config.codex_app_path) / "Contents/MacOS/Codex"),
        "codex app-server",
    ]
    lines: list[str] = []
    for pattern in patterns:
        try:
            result = subprocess.run(
                ["pgrep", "-af", pattern],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            continue
        for line in result.stdout.splitlines():
            if line and line not in lines:
                lines.append(line)
    return lines


def open_codex_desktop(config) -> dict[str, Any]:
    subprocess.run(["open", str(config.codex_app_path)], check=False)
    return get_codex_status(config)


def restart_codex_desktop(config) -> dict[str, Any]:
    subprocess.run(["pkill", "-x", "Codex"], check=False)
    time.sleep(1.0)
    return open_codex_desktop(config)


def get_codex_status(config) -> dict[str, Any]:
    config_path = Path(config.codex_config_path)
    state = load_state(config)
    current_model = None
    current_provider = None
    provider_configured = False
    parse_error = None

    try:
        doc = _read_toml(config_path)
        current_model = doc.get("model")
        current_provider = doc.get("model_provider")
        provider_configured = config.codex_provider_id in doc.get("model_providers", {})
    except Exception as e:
        parse_error = str(e)

    process_lines = _codex_process_lines(config)
    return {
        "codex_home": str(config.codex_home),
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "config_parse_error": parse_error,
        "managed_state_path": str(_state_path(config)),
        "managed_state_exists": state is not None,
        "provider_id": config.codex_provider_id,
        "current_model": current_model,
        "current_model_provider": current_provider,
        "provider_configured": provider_configured,
        "is_applied": current_model == config.default_model
        and current_provider == config.codex_provider_id
        and provider_configured,
        "proxy_token_file": str(config.proxy_token_file),
        "proxy_token_file_exists": Path(config.proxy_token_file).exists(),
        "codex_app_path": str(config.codex_app_path),
        "codex_app_exists": Path(config.codex_app_path).exists(),
        "codex_running": bool(process_lines),
        "codex_processes": process_lines,
    }


def maybe_apply_and_launch(config, log_manager=None) -> None:
    """Apply config and launch desktop according to startup flags."""
    if config.auto_apply_codex_config:
        status = apply_codex_config(config)
        if log_manager:
            log_manager.log_server_event("info", "Codex config applied", status)

    if config.auto_restart_codex_desktop:
        status = restart_codex_desktop(config)
        if log_manager:
            log_manager.log_server_event("info", "Codex desktop launch requested", status)
