# Keras Implementation

While the Huggingface implementation allows us to train models of all sizes,
their deployment is mostly server-only.

For on device models, it might be more beneficial to use the Keras implementation, and train a LORA for gemini-nano:
https://ai.google.dev/gemini-api/docs/get-started/android_aicore

It is not yet clear how to train a LORA model for Gemini Nano, and so we start with a Gemma model.