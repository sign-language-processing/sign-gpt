import gzip
import json
from pathlib import Path

import tensorflow_datasets as tfds
from sign_language_datasets.datasets.dgs_corpus import DgsCorpusConfig
from tqdm import tqdm

from sign_gpt.custom_datasets.dataset_utils import format_task
from sign_gpt.language_utils.i18n import i18n

DATASET_NAME = "dgs_corpus"
DATA_PATH = Path(__file__).parent.parent.parent / "processed" / DATASET_NAME
DATA_PATH.mkdir(parents=True, exist_ok=True)

config = DgsCorpusConfig(name="only-annotations-sentence-level-uzh", version="1.0.0",
                         include_video=False, include_pose=None, data_type="sentence",
                         split="3.0.0-uzh-sentence")
dataset = tfds.load(name=DATASET_NAME, builder_kwargs={"config": config})


TASKS = {
    "gloss_to_text": {
        "system": "Given a sequence of {signed_language}, {spoken_language} glosses and (mouthings) following the style and conventions of The DGS Corpus, translate it into a natural {spoken_language} sentence.",
        "messages": [{
            "input": "{gloss}",
            "output": "{text}",
        }]
    },
    "text_to_gloss": {
        "system": "Given a {spoken_language} sentence, convert it into a sequence of {signed_language}, {spoken_language} glosses and (mouthings) following the style and conventions of The DGS Corpus.",
        "messages": [{
            "input": "{text}",
            "output": "{gloss}",
        }]
    }
}


def build_gloss_text(glosses: list[str],
                     gloss_starts: list[int],
                     mouthings: list[str],
                     mouthings_start: list[int]) -> str:
    text = ""
    mouthing_index = 0
    for gloss, gloss_start in zip(glosses, gloss_starts):
        text += " " + gloss
        gloss_mouthings = []
        while mouthing_index < len(mouthings) and mouthings_start[mouthing_index] <= gloss_start:
            gloss_mouthings.append(mouthings[mouthing_index])
            mouthing_index += 1
        if len(gloss_mouthings) > 0:
            text += f" ({' '.join(gloss_mouthings)})"
    return text.strip()


for split, split_data in dataset.items():
    split_files = {
        task: gzip.open(DATA_PATH / f"{task}.{split}.jsonl.gz", "wt", encoding="utf-8")
        for task in TASKS
    }

    for datum in tqdm(split_data):
        sentence = datum['sentence']

        glosses = sentence['glosses']
        german_glosses = [g.numpy().decode('utf-8') for g in glosses['Gebärde']]
        english_glosses = [g.numpy().decode('utf-8') for g in glosses['Sign']]
        gloss_start_times = [int(t.numpy()) for t in glosses['start']]
        gloss_end_times = [int(t.numpy()) for t in glosses['end']]

        mouthings = sentence["mouthings"]
        mouthings_text = [m.numpy().decode('utf-8') for m in mouthings['mouthing']]
        mouthings_start_times = [int(t.numpy()) for t in mouthings['start']]
        mouthings_end_times = [int(t.numpy()) for t in mouthings['end']]

        params_list = [
            {
                "gloss": build_gloss_text(german_glosses, gloss_start_times, mouthings_text, mouthings_start_times),
                "text": sentence['german'].numpy().decode('utf-8'),
                "signed_language": i18n("signed_languages", "gsg"),
                "spoken_language": i18n("languages", "de"),
            },
            {
                "gloss": build_gloss_text(english_glosses, gloss_start_times, mouthings_text, mouthings_start_times),
                "text": sentence['english'].numpy().decode('utf-8'),
                "signed_language": i18n("signed_languages", "gsg"),
                "spoken_language": i18n("languages", "en"),
            }
        ]

        for params in params_list:
            # Some English text is missing, and should not be included in the dataset
            if params["text"].strip() == "":
                print("Skipping", params)
                continue
            for task, file in split_files.items():
                file.write(json.dumps(format_task(TASKS[task], params)) + "\n")

    for file in split_files.values():
        file.close()
