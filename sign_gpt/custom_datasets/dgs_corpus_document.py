import gzip
import json
from pathlib import Path

import tensorflow_datasets as tfds
from sign_language_datasets.datasets.dgs_corpus import DgsCorpusConfig
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences
from tqdm import tqdm

from sign_gpt.custom_datasets.dataset_utils import format_long_task
from sign_gpt.custom_datasets.dgs_corpus import TASKS, build_gloss_text
from sign_gpt.language_utils.i18n import i18n

DATASET_NAME = "dgs_corpus_document"
DATA_PATH = Path(__file__).parent.parent.parent / "processed" / DATASET_NAME
DATA_PATH.mkdir(parents=True, exist_ok=True)

config = DgsCorpusConfig(name="only-annotations-document-level-uzh", version="1.0.0",
                         include_video=False, include_pose=None, data_type="document",
                         split="3.0.0-uzh-document")
dataset = tfds.load(name="dgs_corpus", builder_kwargs={"config": config})

languages = {
    ("en", "English"): "Sign",
    ("de", "German"): "Gebärde"
}

for split, split_data in dataset.items():
    split_files = {
        task: gzip.open(DATA_PATH / f"{task}.{split}.jsonl.gz", "wt", encoding="utf-8")
        for task in TASKS
    }

    for datum in tqdm(split_data):
        elan_path = datum["paths"]["eaf"].numpy().decode('utf-8')
        sentences = list(get_elan_sentences(elan_path))

        for (lang_code, lang_name), gloss_key in languages.items():
            all_messages = []
            for sentence in sentences:
                glosses = [g[gloss_key] for g in sentence['glosses']]
                gloss_start_times = [g['start'] for g in sentence['glosses']]
                gloss_end_times = [g['end'] for g in sentence['glosses']]
                mouthings_text = [m['mouthing'] for m in sentence['mouthings']]
                mouthings_start_times = [m['start'] for m in sentence['mouthings']]

                params = {
                    "gloss": build_gloss_text(glosses, gloss_start_times, mouthings_text, mouthings_start_times),
                    "text": sentence[lang_name.lower()],
                    "signed_language": i18n("signed_languages", "gsg"),
                    "spoken_language": i18n("languages", lang_code),
                }
                if params["text"] is None:
                    print("Skipping because text is None", params)
                    continue
                if params["text"].strip() == "":
                    print("Skipping because text is empty", params)
                    continue
                if params["gloss"].strip() == "":
                    print("Skipping because gloss is empty", params)
                    continue
                all_messages.append(params)

            if len(all_messages) > 0:
                for task, file in split_files.items():
                    file.write(json.dumps(format_long_task(TASKS[task], all_messages)) + "\n")

    for file in split_files.values():
        file.close()
