# Chinese Olfactory Corpus (LLM-assisted Annotation)

This repository provides code for constructing a Chinese olfactory corpus using
large language model (LLM)–assisted named entity recognition (NER).

## Annotation Method
Annotations are generated using the Gemini 2.0 Flash model with a fixed,
explicitly defined prompt specifying entity categories and output format.
The full prompt is embedded verbatim in the source code.

### Code Usage

Before running the training script, please specify:

- `EXCEL_FOLDER`: path to the folder containing the olfactory corpus Excel files
- `PRETRAINED_MODEL`: a HuggingFace model name or a local pretrained model path

Example:
```python
EXCEL_EXCEL = "./data/Chinese_Olfactory_Corpus"
PRETRAINED_MODEL = "bert-base-chinese"

