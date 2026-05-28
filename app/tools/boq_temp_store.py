import os
import threading
from pathlib import Path
from typing import Optional


_lock = threading.Lock()
_latest_csv_by_user: dict[str, str] = {}
_temp_dir = Path("data") / "temp" / "boq"


def normalize_user_id_for_filename(user_id: str) -> str:
    """
    Build a safe filename fragment for a user id.
    """
    user_id = (user_id or "anonymous").strip() or "anonymous"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in user_id)
    return safe or "anonymous"


def _user_csv_glob(user_id: str) -> str:
    return f"boq_{normalize_user_id_for_filename(user_id)}_*.csv"


def _delete_user_csv_files(user_id: str) -> None:
    if not _temp_dir.exists():
        return

    for path in _temp_dir.glob(_user_csv_glob(user_id)):
        try:
            if path.exists():
                path.unlink()
        except Exception:
            # best-effort cleanup; do not break request processing
            pass


def set_latest_csv(user_id: str, csv_path: str) -> None:
    """
    Store latest BOQ CSV for a user and delete the previous temp file (if any).
    """
    user_id = (user_id or "anonymous").strip() or "anonymous"
    with _lock:
        old = _latest_csv_by_user.get(user_id)
        _latest_csv_by_user[user_id] = csv_path

    if old and old != csv_path:
        try:
            if os.path.exists(old):
                os.remove(old)
        except Exception:
            # best-effort cleanup; do not break request processing
            pass


def get_latest_csv(user_id: str) -> Optional[str]:
    user_id = (user_id or "anonymous").strip() or "anonymous"
    with _lock:
        return _latest_csv_by_user.get(user_id)


def clear_latest_csv(user_id: str) -> None:
    user_id = (user_id or "anonymous").strip() or "anonymous"
    with _lock:
        old = _latest_csv_by_user.pop(user_id, None)

    _delete_user_csv_files(user_id)

    if old:
        try:
            if os.path.exists(old):
                os.remove(old)
        except Exception:
            pass

