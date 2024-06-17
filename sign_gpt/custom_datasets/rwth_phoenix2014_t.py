import json
import gzip
from pathlib import Path

import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig

DATASET_NAME = "rwth_phoenix2014_t"
DATA_PATH = Path(f"processed/{DATASET_NAME}")
DATA_PATH.mkdir(parents=True, exist_ok=True)

config = SignDatasetConfig(name="only-annotations", version="3.0.0", include_video=False)
dataset = tfds.load(name=DATASET_NAME, builder_kwargs=dict(config=config))

TASKS = {
    "gloss_to_text": "Given a sequence of German Sign Language glosses following the style and conventions of the RWTH-PHOENIX-Weather 2014 T dataset, convert it into a natural German sentence, lowercased.\nInput: {gloss}\nOutput: {text}",
    "text_to_gloss": "Given a German sentence, convert it into a sequence of German Sign Language glosses following the style and conventions of the RWTH-PHOENIX-Weather 2014 T dataset.\nInput: {text}\nOutput: {gloss}",
}

for split, split_data in dataset.items():
    split_files = {
        task: gzip.open(DATA_PATH / f"{task}.{split}.jsonl.gz", "wt", encoding="utf-8")
        for task in TASKS
    }

    for datum in split_data:
        params = {
            "gloss": datum['gloss'].numpy().decode('utf-8'),
            "text": datum['text'].numpy().decode('utf-8')
        }
        for task, file in split_files.items():
            instruction_text = TASKS[task].format(**params)
            file.write(json.dumps({"text": instruction_text}) + "\n")

    for file in split_files.values():
        file.close()
