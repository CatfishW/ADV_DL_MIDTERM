
from tqdm import tqdm
from transformers import T5Tokenizer
from model.model import model,tokenizer
from dataset.dataset import test_dataset,compute_metrics,data_collator,tokenized_books,tokenized_books_eval
from config.config import cfg
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from colorama import Fore, Style, init
#SET CUDA AVAILABLE DEVICES
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
inputs,targets=[],[]
for _, examples in tqdm(zip(range(200), iter(test_dataset)), total=200):
    inputs.append(examples["translation"]['en'])
    targets.append(examples["translation"]['fr'])
model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
print(model_inputs.keys())
'''
dict_keys(['input_ids', 'attention_mask', 'labels'])
'input_ids': [[101, 278...]] These IDs represent the tokenized form of the input text.
'attention_mask': [[1, 1...]] The attention mask is used to indicate which tokens should be attended to (1) and which should be ignored (0).
'labels': [[22833,3,...]] For supervised learning tasks.
'''
if cfg['model']['train_loop_type'] == "huggingface":
    training_args = Seq2SeqTrainingArguments(
        output_dir="ckpt",
        learning_rate=1e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.001,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True, #change to bf16=True for XPU
        push_to_hub=False,
        resume_from_checkpoint=cfg['model']['resume'],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_books,
        eval_dataset=tokenized_books_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    #Metrics Computations
    results = trainer.predict(tokenized_books_eval)
    '''
    PredictionOutput(predictions=array([[    0,  1064,   285, ...,    87,   287, 19882],
       [    0,  3557,   210, ...,     0,     0,     0],
       [    0, 17129,  5545, ...,     3, 26375,   245],
       ...,
       [    0,  1955,   276, ...,     0,     0,     0],
       [    0,  3039,    73, ...,    15,    20,  2143],
       [    0,  9236,     9, ...,     0,     0,     0]]), 
       label_ids=array([[ 7227,   142,  8063, ...,    15,     5,     1],
       [ 3557,   210,     3, ...,  -100,  -100,  -100],
       [  622,     3, 29725, ...,  -100,  -100,  -100],
       ...,
       [ 1955,   276, 12220, ...,  -100,  -100,  -100],
       [ 3039,   197, 29068, ...,  -100,  -100,  -100],
       [  312, 26274,   146, ...,  -100,  -100,  -100]]), 
       metrics={'test_loss': 1.4194819927215576, 
       'test_bleu': 4.7773, 'test_gen_len': 17.4118, 
       'test_runtime': 3.1035, 'test_samples_per_second': 65.733, 
       'test_steps_per_second': 4.189})
    '''
    print('test_bleu:',results.metrics['test_bleu'])
    '''
    BLEU (Bilingual Evaluation Understudy) is a metric ranging from 0 to 100 
    for evaluating the quality of text which has been 
    machine-translated from one language to another.
    '''
    #map the predictions to the actual words
    decoded_preds = tokenizer.batch_decode(results.predictions, skip_special_tokens=True)
    for i in range(5):
        print(Fore.GREEN + "English Input:")
        print(Fore.BLUE + tokenized_books_eval['translation'][i]['en'])
        print(Fore.GREEN + "French Prediction:")
        print(Fore.BLUE + decoded_preds[i])
        print(Fore.RED + "Actual French Translation:")
        print(Fore.RESET + tokenized_books_eval['translation'][i]['fr'])
        print("==========================================================================================================")



