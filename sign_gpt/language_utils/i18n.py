import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def load_file(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def i18n(i18n_type: str, key: str):
    i18n_path = Path(__file__).parent / f"i18n/{i18n_type}.json"
    data = load_file(i18n_path)
    return data[key] if key in data else key
