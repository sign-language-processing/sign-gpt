import os
from pathlib import Path

from datasets import load_dataset


def load_split_data(split, mapping_fn):
    processed_dir = Path(__file__).parent.parent.parent / 'processed'
    data_files = str(processed_dir / '**' / f'*.{split}.jsonl.gz')
    dataset = load_dataset('json', data_files=data_files, split='train')
    return dataset.map(mapping_fn, num_proc=os.cpu_count())


def test_mapping_fn(datum, mapping_fn):
    # Remove the last output from the mapping function
    datum["messages"][-1]["output"] = ""
    return mapping_fn(datum)


def load_datasets(mapping_fn):
    # Load train, dev, and test splits
    train_dataset = load_split_data('train', mapping_fn)
    validation_dataset = load_split_data('validation', mapping_fn)
    test_dataset = load_split_data('test', lambda datum: test_mapping_fn(datum, mapping_fn))

    return train_dataset, validation_dataset, test_dataset
