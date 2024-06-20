import gzip
import json
from pathlib import Path

import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig

from sign_gpt.custom_datasets.dataset_utils import format_task

DATASET_NAME = "rwth_phoenix2014_t"
DATA_PATH = Path(__file__).parent.parent.parent / "processed" / DATASET_NAME
DATA_PATH.mkdir(parents=True, exist_ok=True)

config = SignDatasetConfig(name="only-annotations", version="3.0.0", include_video=False)
dataset = tfds.load(name=DATASET_NAME, builder_kwargs=dict(config=config))

TASKS = {
    "gloss_to_text": {
        "system": "Given a sequence of German Sign Language glosses following the style and conventions of the RWTH-PHOENIX-Weather 2014 T dataset, convert it into a natural German sentence, lowercased.",
        "messages": [{
            "input": "{gloss}",
            "output": "{text}",
        }]
    },
    "text_to_gloss": {
        "system": "Given a German sentence, convert it into a sequence of German Sign Language glosses following the style and conventions of the RWTH-PHOENIX-Weather 2014 T dataset.",
        "messages": [{
            "input": "{text}",
            "output": "{gloss}",
        }]
    }
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
            file.write(json.dumps(format_task(TASKS[task], params)) + "\n")

    for file in split_files.values():
        file.close()
