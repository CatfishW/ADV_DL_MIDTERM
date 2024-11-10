from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM
from config.config import cfg
from model.model import model,tokenizer
from dataset.dataset import compute_metrics,data_collator,tokenized_books,tokenized_books_eval,tokenized_books_test
from dataset.dataset import test_dataset
#SET CUDA AVAILABLE DEVICES
import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = cfg['model']['CUDA_VISIBLE_DEVICES']
# inputs,targets=[],[]
# for _, examples in tqdm(zip(range(200), iter(test_dataset)), total=200):
#     inputs.append(examples["translation"]['en'])
#     targets.append(examples["translation"]['fr'])
# model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
# print('model_inputs:')
# print(model_inputs.keys())
'''
dict_keys(['input_ids', 'attention_mask', 'labels'])
'input_ids': [[101, 278...]] These IDs represent the tokenized form of the input text.
'attention_mask': [[1, 1...]] The attention mask is used to indicate which tokens should be attended to (1) and which should be ignored (0).
'labels': [[22833,3,...]] For supervised learning tasks.
'''
if cfg['model']['train_loop_type'] == "huggingface":
    training_args = Seq2SeqTrainingArguments(
        output_dir="ckpt",
        learning_rate=cfg['train']['learning_rate'],
        per_device_train_batch_size=cfg['train']['per_device_train_batch_size'],
        per_device_eval_batch_size=cfg['train']['per_device_eval_batch_size'],
        weight_decay=0.001,
        save_total_limit=3,
        num_train_epochs=cfg['train']['num_train_epochs'],
        predict_with_generate=True,
        fp16=True, #change to bf16=True for XPU
        push_to_hub=False,
        do_eval=cfg['model']['do_eval'],
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg['model']['resume']) if cfg['model']['resume'] is not None else None
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_books,
        eval_dataset=tokenized_books_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
else:
    #add text of to be implemented on Error
    raise NotImplementedError("Only huggingface train loop is implemented for now. I'll implement the pytorch train loops for the final project. :)")