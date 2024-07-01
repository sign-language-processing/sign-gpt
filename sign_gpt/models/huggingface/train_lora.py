import argparse
import os

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling

from sign_gpt.models.data import load_datasets

os.environ["WANDB_PROJECT"] = "sign-gpt"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


def prep_llama3_instruction(tokenizer):
    def prep_instruction(datum):
        # From https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
        text = "<|start_header_id|>system<|end_header_id|>\n\n" + datum['system'] + "<|eot_id|>"
        for message in datum['messages']:
            text += "<|start_header_id|>user<|end_header_id|>\n\n" + message['input'] + "<|eot_id|>"
            text += "<|start_header_id|>assistant<|end_header_id|>\n\n" + message['output'] + "<|eot_id|>"

        tokens = tokenizer(text, padding='max_length', truncation=True, max_length=2048)

        return {
            "text": text,
            "system": None,
            "messages": None,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]

        }

    return prep_instruction


def load_tokenizer(model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = 0  # unclear if/why this is needed
    return tokenizer


def load_model(model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
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

    return model


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
        eval_strategy="epoch",
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

    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    trainer.train()

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, help="Path to output directory")
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    train_dataset, validation_dataset, test_dataset = load_datasets(prep_llama3_instruction(tokenizer))
    train_dataset = train_dataset.shuffle(seed=42)
    print(train_dataset["text"][0])

    model = load_model()

    trainer = get_trainer(model, tokenizer, train_dataset, validation_dataset, args.output_dir)
    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
