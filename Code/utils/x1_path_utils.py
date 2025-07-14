from pathlib import Path
import os

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def resolve_path(p: str | os.PathLike) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else repo_root() / p