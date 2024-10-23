import gzip
import json
from pathlib import Path

import signwriting
from signwriting.formats.fsw_to_swu import fsw2swu

from sign_gpt.custom_datasets.dataset_utils import format_task
from sign_gpt.language_utils.i18n import i18n

DATASET_NAME = "signwriting_hamnosys"
DATA_PATH = Path(__file__).parent.parent.parent / "processed" / DATASET_NAME
DATA_PATH.mkdir(parents=True, exist_ok=True)

TASKS = {
    "hamnosys_to_signwriting": {
        "system": "Given a sequence of HamNoSys notation for a sign in {signed_language}, translate it into SignWriting.",
        "messages": [{
            "input": "{hamnosys}",
            "output": "{signwriting}",
        }]
    },
    "signwriting_to_hamnosys": {
        "system": "Given a SignWriting notation for a sign in {signed_language}, translate it into HamNoSys.",
        "messages": [{
            "input": "{signwriting}",
            "output": "{hamnosys}",
        }]
    }
}

# find the file /hamnosys/parallel.json in the signwriting package
parallel_file = Path(signwriting.__file__).parent / "hamnosys" / "parallel.json"

with open(parallel_file, "r", encoding="utf-8") as f:
    data = json.load(f)

split_files = {
    task: gzip.open(DATA_PATH / f"{task}.train.jsonl.gz", "wt", encoding="utf-8")
    for task in TASKS
}

for datum in data:
    if "signwriting" in datum and "hamnosys" in datum and len(datum["hamnosys"]) == 1:
        params = {
            "hamnosys": datum['hamnosys'][0],
            "signwriting": fsw2swu(datum['signwriting']),
            "signed_language": i18n("signed_languages", "sgg"),
        }

        for task, file in split_files.items():
            file.write(json.dumps(format_task(TASKS[task], params)) + "\n")

for file in split_files.values():
    file.close()
