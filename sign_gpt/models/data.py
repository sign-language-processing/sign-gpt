import os
from pathlib import Path

from datasets import load_dataset


def load_split_data(split, mapping_fn):
    processed_dir = Path(__file__).parent.parent.parent / 'processed'
    data_files = str(processed_dir / '*' / f'*.{split}.jsonl.gz')
    dataset = load_dataset('json', data_files=data_files, split='train')
    return dataset.map(mapping_fn, num_proc=os.cpu_count())


def dataset_test_mapping_fn(datum, mapping_fn):
    # Remove the last output from the mapping function
    datum["messages"][-1]["output"] = ""
    return mapping_fn(datum)


def load_datasets(mapping_fn, hide_test=True):
    # Load train, dev, and test splits
    train_dataset = load_split_data('train', mapping_fn)
    validation_dataset = load_split_data('validation', mapping_fn)

    test_mapping = (lambda datum: dataset_test_mapping_fn(datum, mapping_fn)) if hide_test else mapping_fn
    test_dataset = load_split_data('test', test_mapping)

    return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    def check_dataset(datum):
        for message in datum["messages"]:
            if message["input"] == "" or message["output"] == "":
                print(datum)
                raise Exception("Empty input or output")

    load_datasets(check_dataset, hide_test=False)
