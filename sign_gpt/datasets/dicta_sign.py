import gzip
import json
from pathlib import Path

import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig

from sign_gpt.language_utils.i18n import i18n
from sign_gpt.language_utils.info import sign_language_by_abbreviation

DATASET_NAME = "dicta_sign"
DATA_PATH = Path(f"processed/{DATASET_NAME}")
DATA_PATH.mkdir(parents=True, exist_ok=True)

config = SignDatasetConfig(name="only-annotations", version="1.0.0", include_video=False, include_pose=None)
dataset = tfds.load(name=DATASET_NAME, builder_kwargs=dict(config=config))

TASKS = {
    "hamnosys_to_text": "Given a sequence of HamNoSys notation for a sign in {signed_language}, translate it to {spoken_language} text.\nInput: {hamnosys}\nOutput: {text}",
    "text_to_hamnosys": "Given a sequence of {spoken_language} text, translate it to HamNoSys notation in {signed_language}.\nInput: {text}\nOutput: {hamnosys}",
}

for split, split_data in dataset.items():
    split_files = {
        task: gzip.open(DATA_PATH / f"{task}.{split}.jsonl.gz", "wt", encoding="utf-8")
        for task in TASKS
    }

    for datum in split_data:
        sign_language_abbr = datum['signed_language'].numpy().decode('utf-8')
        sign_language_key = sign_language_by_abbreviation(sign_language_abbr)
        spoken_language_key = datum['spoken_language'].numpy().decode('utf-8')
        params = {
            "hamnosys": datum['hamnosys'].numpy().decode('utf-8'),
            "text": datum['text'].numpy().decode('utf-8'),
            "signed_language": i18n("signed_languages", sign_language_key),
            "spoken_language": i18n("languages", spoken_language_key),
            "gloss": datum['gloss'].numpy().decode('utf-8'),
        }
        for task, file in split_files.items():
            instruction_text = TASKS[task].format(**params)
            file.write(json.dumps({"text": instruction_text}) + "\n")

    for file in split_files.values():
        file.close()
