"""Atomic JSON records shared by dataset preparation and experiment runs."""

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as temp:
        tmp = Path(temp.name)
        try:
            json.dump(data, temp, indent=2, allow_nan=False)
            temp.write("\n")
            temp.flush()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def fingerprint(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
