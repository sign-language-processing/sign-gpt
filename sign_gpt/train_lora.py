import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling

os.environ["WANDB_PROJECT"] = "sign-gpt"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


def prep_llama3_instruction(datum):
    text = datum['text']
    text = "<|start_header_id|>system<|end_header_id|>\n\n" + text + "<|eot_id|>"
    text = text.replace("\nInput: ", "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n")
    text = text.replace("\nOutput: ", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    return {"text": text}


def load_split_data(split):
    processed_dir = Path(__file__).parent.parent / 'processed'
    data_files = str(processed_dir / '**' / f'*.{split}.jsonl.gz')
    dataset = load_dataset('json', data_files=data_files, split='train')
    return dataset.map(prep_llama3_instruction, num_proc=os.cpu_count())


def load_datasets():
    # Load train, dev, and test splits
    train_dataset = load_split_data('train')
    validation_dataset = load_split_data('validation')
    test_dataset = load_split_data('test')

    return train_dataset, validation_dataset, test_dataset


def load_model(model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = 0  # unclear if/why this is needed
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    return model, tokenizer


def get_trainer(model, tokenizer, train_dataset, validation_dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        bf16=True,
        learning_rate=3e-4,
        num_train_epochs=10,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        optim="adafactor",
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        ddp_find_unused_parameters=None,
        auto_find_batch_size=True,
        report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # compute_metrics=compute_metrics, # TODO
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # TODO: unclear why or if this is needed
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    trainer.train()

    return trainer


if __name__ == "__main__":
    output_dir = "/tmp"

    train_dataset, validation_dataset, test_dataset = load_datasets()
    print(train_dataset["text"][-1])

    model, tokenizer = load_model()

    trainer = get_trainer(model, tokenizer, train_dataset, validation_dataset, output_dir)
    trainer.train()

    trainer.model.save_pretrained(output_dir)
