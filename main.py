
from tqdm import tqdm
from config.config import cfg
from colorama import Fore, Style, init
from engine import trainer
from dataset.dataset import tokenized_books_eval,tokenized_books_test
from model.model import tokenizer
if not cfg['model']['do_eval']:
    #run test dataset every 1000 steps
    trainer.train()
trainer.evaluate(eval_dataset=tokenized_books_test)
#Metrics Computations on Eval Dataset
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
print(Fore.YELLOW + "Metrics:")
print('TEST SET BLEU:',results.metrics['test_bleu'])



