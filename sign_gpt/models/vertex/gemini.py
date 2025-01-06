import vertexai
from vertexai.preview.tuning import sft

from sign_gpt.models.data import load_datasets


def prep_gemini_instruction(datum):
    messages = []
    for message in datum['messages']:
        messages.append({"role": "user", "parts": [{"text": message['input']}]})
        messages.append({"role": "model", "parts": [{"text": message['output']}]})

    if any(message["parts"][0]["text"] == "" for message in messages):
        print(messages)
        raise ValueError("Empty input or output will make gemini crash")

    return {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": datum['system']}]
        },
        "contents": messages
    }


def main():
    gcs_path = "gcs://sign-language-datasets/public/sign-gpt"
    gs_path = "gs://sign-language-datasets/public/sign-gpt"

    # Upload datasets to GCS
    train_dataset, validation_dataset, test_dataset = load_datasets(prep_gemini_instruction, hide_test=False,
                                                                    remove_columns=("system", "messages"))

    if len(train_dataset) > 50_000:
        raise ValueError("Training dataset is too large for tuning with Vertex AI. "
                         "max allowed training steps is 50,000, with an unknown batch size.")

    train_dataset.to_json(gcs_path + "/train.jsonl", lines=True)
    # Validation Dataset can not contain more than 256 model turns.
    validation_dataset = validation_dataset.select(range(256))
    validation_dataset.to_json(gcs_path + "/validation.jsonl", lines=True)
    test_dataset.to_json(gcs_path + "/test.jsonl", lines=True)

    # Create training job
    vertexai.init(project="sign-mt", location="us-central1")
    sft_tuning_job = sft.train(
        source_model="gemini-1.5-pro-002",
        train_dataset=gs_path + "/train.jsonl",
        validation_dataset=gs_path + "/validation.jsonl",
        epochs=1,
        learning_rate_multiplier=1.0,
        tuned_model_display_name="tuned_gemini_pro",
    )


if __name__ == "__main__":
    main()
