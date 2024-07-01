"""Based on https://ai.google.dev/gemma/docs/lora_tuning"""

import os

os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

from sign_gpt.models.data import load_datasets
import wandb

import keras
import keras_nlp
from wandb.integration.keras import WandbMetricsLogger

keras.mixed_precision.set_global_policy('mixed_bfloat16')


def prep_gemini_instruction(datum):
    text = "Instruction:\n" + datum['system'] + "\n\n"
    for message in datum['messages']:
        text += "Input:\n" + message['input'] + "\n\n"
        text += "Output:\n" + message['output'] + "\n\n"
    return {"text": text}


def load_model(model_id="gemma_2b_en"):
    model = keras_nlp.models.GemmaCausalLM.from_preset(model_id)

    # Enable LoRA for the model and set the LoRA rank to 4.
    model.backbone.enable_lora(rank=4)

    model.summary()

    # Limit the input sequence length to 1024 (to control memory usage).
    model.preprocessor.sequence_length = 1024
    # Use AdamW (a common optimizer for transformer models).
    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
    )
    # Exclude layernorm and bias terms from decay.
    optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


if __name__ == "__main__":
    output_dir = "/tmp"

    train_dataset, validation_dataset, test_dataset = load_datasets(prep_gemini_instruction)
    print(train_dataset["text"][0])

    model = load_model()

    # Initialize a new W&B run
    wandb.init(project="sign-gpt-gemini")

    model.fit(train_dataset,
              validation_data=validation_dataset,
              callbacks=[WandbMetricsLogger()],
              epochs=1,
              batch_size=1  # TODO: check how much fits on GPU
              )
