import torch
import datasets
from datasets import load_dataset
from model.model import tokenizer
from colorama import Fore, Style, init
from config.config import cfg
if cfg['data']['train_type'] == 'partial':
    ri = (datasets.ReadInstruction('train', to=95, unit='%'))
    train_dataset = load_dataset("IWSLT/ted_talks_iwslt", language_pair=("en", "fr"), year="2014", split=ri)
elif cfg['data']['train_type'] == 'full':
    train_dataset = load_dataset("IWSLT/ted_talks_iwslt", language_pair=("en", "fr"), year="2014")
if cfg['data']['test_type'] == 'test_0_50pct_ds':
    ri = (datasets.ReadInstruction('train', to=50, unit='%'))
    test_dataset = load_dataset("IWSLT/ted_talks_iwslt", language_pair=("en", "fr"), year="2016", split=ri)
elif cfg['data']['test_type'] == 'test_95_100pct_ds':
    ri = (datasets.ReadInstruction('train', from_=95, unit='%'))
    test_dataset = load_dataset("IWSLT/ted_talks_iwslt", language_pair=("en", "fr"), year="2016", split=ri)
else:
    raise NotImplementedError
if cfg['data']['data_increment'] > 0:
    ...


#=====Download and Load the dataset=====
#ri = (datasets.ReadInstruction('train', to=50, unit='%'))
#train_0_50pct_ds = load_dataset("IWSLT/ted_talks_iwslt", language_pair=("en", "fr"), year="2016", split=ri)
'''
train_0_50pct_ds: Dataset({
    features: ['translation'],
    num_rows: 2035
})
'''
#ri = (datasets.ReadInstruction('train', from_=95, unit='%'))
#test_95_100pct_ds = load_dataset("IWSLT/ted_talks_iwslt", language_pair=("en", "fr"), year="2016", split=ri)
#print(test_95_100pct_ds)
'''
test_95_100pct_ds: Dataset({
    features: ['translation'],
    num_rows: 204
})
'''
#Difference:
'''
train_0_50pct_ds dataset:
    Uses the instruction datasets.ReadInstruction('train', to=50, unit='%').
    Includes the first 50% of the training dataset.
test_95_100pct_ds dataset:
    Uses the instruction datasets.ReadInstruction('train', from_=95, unit='%').
    Includes the last 5% of the training dataset.
This means train_0_50pct_ds contains the first half of the training dataset, while test_95_100pct_ds contains the last small portion of the training dataset.
'''
import numpy as np
import evaluate
from transformers import DataCollatorForSeq2Seq
def preprocess_function(examples):
    #inputs = [prefix + example[source_lang] for example in examples["translation"]]
    inputs = [example[cfg['data']['source_lang']] for example in examples["translation"]]
    targets = [example[cfg['data']['target_lang']] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

metric = evaluate.load("sacrebleu")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=cfg['model']['name'])
tokenized_books = train_dataset.map(preprocess_function, batched=True)
tokenized_books_eval = test_dataset.map(preprocess_function, batched=True)
#print train_dataset size and test_dataset size
print(Fore.GREEN + "Train Dataset Size:")
print(Fore.BLUE + str(len(tokenized_books['translation'])))
print(Fore.GREEN + "Test Dataset Size:")
print(Fore.BLUE + str(len(tokenized_books_eval['translation'])))
print("==========================================================================================================")
# Initialize colorama
init(autoreset=True)
# Show some pairing examples of the data
print(Fore.GREEN + "Data Example 1:")
print(Fore.BLUE + 'language:', cfg['data']['source_lang'], '', Fore.RESET + tokenized_books_eval['translation'][0][cfg['data']['source_lang']][:30])
print(Fore.RED + 'language:', cfg['data']['target_lang'], '', Fore.RESET + tokenized_books_eval['translation'][0][cfg['data']['target_lang']][:30])
print(Fore.GREEN + "Data Example 2:")
print(Fore.BLUE + 'language:', cfg['data']['source_lang'], '', Fore.RESET + tokenized_books_eval['translation'][1][cfg['data']['source_lang']][:30])
print(Fore.RED + 'language:', cfg['data']['target_lang'], '', Fore.RESET + tokenized_books_eval['translation'][1][cfg['data']['target_lang']][:30])
print("==========================================================================================================")

