import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def load_file(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


warnings = set()


def i18n(i18n_type: str, key: str):
    i18n_path = Path(__file__).parent / f"i18n/{i18n_type}.json"
    data = load_file(i18n_path)
    if key == "":
        return "Unknown"
    if key in data:
        return data[key]
    if key.lower() in data:
        return data[key.lower()]
    warning_key = (i18n_type, key)
    if warning_key not in warnings:
        warnings.add(warning_key)
        print(f"Could not find key '{key}' in i18n file '{i18n_path}'")
    return key
