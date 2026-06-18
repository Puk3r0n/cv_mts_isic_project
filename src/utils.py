import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def as_path(value, base=PROJECT_ROOT):
    p = Path(value)
    return p if p.is_absolute() else (Path(base) / p).resolve()


def run_stamp(suffix=None):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = (suffix or "").strip().replace(" ", "_")
    return f"{stamp}_{suffix}" if suffix else stamp


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def _ser(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_ser) + "\n", encoding="utf-8")
    return path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
