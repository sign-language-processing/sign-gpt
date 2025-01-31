# SignGPT

Multi Task GPT model for Sign Language Processing, using spoken language text.

## Pre-Tokenization

- Spoken language text is left as is.
- SignWriting is represented as SignWriting in Unicode (SWU).
- HamNoSys is left as is.
- Poses are tokenized using the [MediaPipe vector quantizer](https://github.com/sign-language-processing/sign-vq).


## Pretraining (Idea)

There are many books and articles on Sign Language Processing, but they are not easily/widely accessible.
Collecting a large dataset of text about sign language can be beneficial for large scale pretraining.

## Tasks

This GPT model is designed to be large, and encode a wide variety of tasks in Sign Language Processing.
The tasks we currently cover are

| Task Name                            | Prompt                                                                                                                                                                      | Response                                                                                      | Data                                                                                                                        | 
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Unconstrained SignWriting Generation | Generate a SignWriting sequence in `American Sign Lagnguage`:                                                                                                               | `𝠃𝤘𝤣񁲡𝣳𝣩񈩧𝤉𝣻 𝠃𝤘𝤧񃊫𝣻𝤕񃊢𝣴𝣼񆇡𝤎𝤂񋛕𝤆𝣦 񏌁𝣢𝤂`                                        | [sign-language-processing/signbank-plus](https://github.com/sign-language-processing/signbank-plus)                         |
| SignWriting to Text Translation      | Translate the following `American Sign Language` SignWriting sequence: `𝠃𝤘𝤣񁲡𝣳𝣩񈩧𝤉𝣻 𝠃𝤘𝤧񃊫𝣻𝤕񃊢𝣴𝣼񆇡𝤎𝤂񋛕𝤆𝣦 񏌁𝣢𝤂` to `English` text:                            | `Hello World`                                                                                 | [sign-language-processing/signbank-plus](https://github.com/sign-language-processing/signbank-plus)                         |
| Text to SignWriting Translation      | Translate the following `English` text: `Hello World` to `American Sign Language` SignWriting:                                                                              | `𝠃𝤘𝤣񁲡𝣳𝣩񈩧𝤉𝣻 𝠃𝤘𝤧񃊫𝣻𝤕񃊢𝣴𝣼񆇡𝤎𝤂񋛕𝤆𝣦 񏌁𝣢𝤂`                                        | [sign-language-processing/signbank-plus](https://github.com/sign-language-processing/signbank-plus)                         |
| Single Sign Description              | Describe how to sign the following `English` text: `Hello` in `American Sign Language`:                                                                                     | `With your dominant hand open, touch your forehead and move your hand away, palm facing out.` | [sign-language-processing/signwriting-description](https://github.com/sign-language-processing/signwriting-description)     |
| SignWriting Description              | Describe the sign represented by this `American Sign Language` SignWriting: `𝠃𝤘𝤣񁲡𝣳𝣩񈩧𝤉𝣻`.                                              | `With your dominant hand open, touch your forehead and move your hand away, palm facing out.` | [sign-language-processing/signwriting-description](https://github.com/sign-language-processing/signwriting-description)     |
| Description to SignWriting           | Generate `American Sign Language` SignWriting from the description of a sign: `With your dominant hand open, touch your forehead and move your hand away, palm facing out.` | `𝠃𝤘𝤣񁲡𝣳𝣩񈩧𝤉𝣻`                                             | [sign-language-processing/signwriting-description](https://github.com/sign-language-processing/signwriting-description)     |
| Gloss to Text Translation            | Translate the following `American Sign Language` gloss sequence: `HELLO WORLD` to `English` text:                                                                           | `Hello World`                                                                                 | DGS Corpus, PHOENIX                                                                                                         |
| Text to Gloss Translation            | Translate the following `English` text: `Hello World` to `American Sign Language` glosses:                                                                                  | `HELLO WORLD`                                                                                 | DGS Corpus, PHOENIX                                                                                                         |
| Unconstrained Pose Generation        | Generate a pose sequence in `American Sign Lagnguage`:                                                                                                                      | `code435 code325 ... code492`                                                                 | [sign-language-processing/sign-vq](https://github.com/sign-language-processing/sign-vq)                                     |
| Pose to SignWriting Transcription    | Transcribe the following `American Sign Language` pose sequence: `code435 code325 ... code492` into SignWriting:                                                            | `𝠃𝤘𝤣񁲡𝣳𝣩񈩧𝤉𝣻`                                                          | [sign-language-processing/signwriting-transcription](https://github.com/sign-language-processing/signwriting-transcription) |
| SignWriting to Pose Animation        | Animate the following `American Sign Language` SignWriting sequence `𝠃𝤘𝤣񁲡𝣳𝣩񈩧𝤉𝣻` into poses:                                                        | `code435 code325 ... code492`                                                                 | [sign-language-processing/signwriting-transcription](https://github.com/sign-language-processing/signwriting-transcription) |
| Pose to Text Translation             | Translate the following `American Sign Language` pose sequence: `code435 code325 ... code492` into `English`:                                                               | `Hello World`                                                                                 | SignTube                                                                                                                    |
| Text to Pose Animation               | Animate the following `English` text `Hello World` into `American Sign Language` poses:                                                                                     | `code435 code325 ... code492`                                                                 | SignTube                                                                                                                    |

## Data Generation

Set up the environment:

```bash
conda create --name sign_gpt python=3.11 -y
conda activate sign_gpt

pip install git+https://github.com/sign-language-processing/datasets.git
pip install mediapipe gdown lxml
```

Generate the data:

```bash
python -m sign_gpt.custom_datasets.rwth_phoenix2014_t
python -m sign_gpt.custom_datasets.dicta_sign
python -m sign_gpt.custom_datasets.dgs_types
python -m sign_gpt.custom_datasets.dgs_corpus
python -m sign_gpt.custom_datasets.signbank_plus
python -m sign_gpt.custom_datasets.signwriting_hamnosys
```

### Crawl Idea

We have very large crawlers (such as CommonCrawl) that can be used to collect data from websites/books.
We can vectorize all videos.
We have very strong and capable language models able to help us create data.
So the idea would be: crawl the web/whatever, feed the contents to a language model to generate a system prompt, 
and the relevant inputs and outputs from the document. We then compile that into "CrawlInstruct"

Videos that include captions are always covnerted to a translation task.

## Model Training

```bash
pip install .[huggingface]
python -m sign_gpt.models.huggingface.train_lora
sbatch sign_gpt/models/huggingface/train.sh

pip install .[keras]
python -m sign_gpt.models.keras.train_gemma
```