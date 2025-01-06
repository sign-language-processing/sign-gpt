import gzip
import json
from pathlib import Path

import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig

from sign_gpt.custom_datasets.dataset_utils import format_task
from sign_gpt.language_utils.i18n import i18n

DATASET_NAME = "dgs_types"
DATA_PATH = Path(__file__).parent.parent.parent / "processed" / DATASET_NAME
DATA_PATH.mkdir(parents=True, exist_ok=True)

config = SignDatasetConfig(name="only-annotations", version="1.0.0",
                           include_video=False, process_video=False, include_pose=None)
dataset = tfds.load(name=DATASET_NAME, builder_kwargs={"config": config})

TASKS = {
    "hamnosys_to_gloss": {
        "system": "Given a sequence of HamNoSys notation for a sign in {signed_language}, translate it into a {spoken_language} gloss according to the DGS Corpus.",
        "messages": [{
            "input": "{hamnosys}",
            "output": "{gloss}",
        }]
    },
    "gloss_to_hamnosys": {
        "system": "Given a {spoken_language} gloss according to the DGS Corpus, translate it into HamNoSys notation in {signed_language}.",
        "messages": [{
            "input": "{gloss}",
            "output": "{hamnosys}",
        }]
    }
}

for split, split_data in dataset.items():
    split_files = {
        task: gzip.open(DATA_PATH / f"{task}.{split}.jsonl.gz", "wt", encoding="utf-8")
        for task in TASKS
    }

    for datum in split_data:
        glosses = [gloss.numpy().decode('utf-8') for gloss in datum['glosses']]
        for gloss in glosses:
            params = {
                "hamnosys": datum['hamnosys'].numpy().decode('utf-8'),
                "gloss": gloss,
                "signed_language": i18n("signed_languages", "gsg"),
                "spoken_language": i18n("languages", "de"),
            }

            if params["gloss"].strip() == "" or params["hamnosys"].strip() == "":
                print("Skipping", params)
                continue

            for task, file in split_files.items():
                file.write(json.dumps(format_task(TASKS[task], params)) + "\n")

    for file in split_files.values():
        file.close()
