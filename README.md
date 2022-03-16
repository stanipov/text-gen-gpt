# Fine tune GPT2 with DeepSpeed to generate LinkedIn-style job postings using just title and keywords

## Intro
It's known that GPT2-like models (GPT2 or GPT Neo) are capable of generation of texts from a string prompt. This method has no means of controlling content of the generated text. Addition of key- and key-phrases extracted from the text allows better control of the generated text.

This small project contains set of scripts to fine tune GPT2 models (L and XL) and generate the exemplary LinkedIn-styled job ads. 

## Fine Tuning procedure
This section describes training data preparation, tuning method and usage of the tuning script.

### Training data description
The fine-tuning is done on job postings on LinkedIn that were scrapped during 2021 with my own scrapper. Approximately 5000 unique job postings were scrapped. The texts were cleaned from artifacts before they were processed to extract 1 - 3 n-grams. The n-grams were extracted using MMR (maximal marginal relevance) of the n-gram and sentence embeddings. The processed data was then split into train-validation sets under condition that the texts contain no more than N tokens and no less than K. The results are locate in **input_data** folder. The files for convenience are pickled Pandas DataFrames.

### Fine tuning
#### Customized tokenizer
As the first step of the fine-tuning a tokenizer is set with addition of separator token. This token will separate the title from the keywords and the training text. For example, a job title *Data Scientist - Recommendation systems* and keywords *Recommendation systems,Python,neural networks,develop algorithm,predict customer interest,TensorFlow,ML frameworks* will be separated by the <|SEP|> token. 

The tokenizer is set to pad any input sequence that is shorter than **max_len** parameter (in a training config) and truncate anything that is longer than this. Due to limits of available VRAM, the texts should be limited in length. Many LinkedIn job postings have lengths well over 1000 tokens. In my experience, texts with 800+ tokens encompass roughly 1/4 of the total dataset of almost 5000 job postings. Even with DeepSpeed and moderate 12GB VRAM GPU this results in OOM errors when a larger models like GPT2 L or XL are tuned. Therefore a simply truncated dataset will contain a substantial fraction of incomplete texts. This will bias the model towards generation of incomplete and broken postings. A better approach is to filter job postings that are less than *N* tokens large. 

 **max_len** should be chosen larger than the maximal number of tokens in the training data to accommodate most of text. Since majority of the text contain fairly common (sometimes even looking similarly) endings on employer being equal opportunity, non-discriminative, etc, the generalization of the tuned model should not be seriously affected. In the same time it allows to feed larger texts into the model that often contain interesting texts.

#### Training
The fin tuning is implemented in ```cli_training.py -cfg config_name``` script. The fine tuning parameters are written in a .json file. The real training configs and DeepSpped (ds) configs are locate in **generate_configs** and **DS_configs** respectively. 
Refer to **NLP_model class** of the **utils.py** file for technical implementations. 

For training in a Jupyter Notebook on a local machine, use these parameters:
```python
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9994' # modify if RuntimeError: Address already in use
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1" 
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```
