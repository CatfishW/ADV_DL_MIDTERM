from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from colorama import Fore
import os
import torch
from config.config import cfg
from model.model import tokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Load model and tokenizer from checkpoints
model_checkpoint_folder = cfg['model']['resume']
model_checkpoint = os.path.join(model_checkpoint_folder)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
while True:
    inputs = input("Enter a sentence in English (or type 'exit' to quit): ")
    if inputs.lower() == 'exit':
        break
    model_inputs = tokenizer(inputs, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**model_inputs)
    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("==========================================================================================================")
    print(Fore.GREEN + "English Input:", inputs)
    print(Fore.YELLOW + "French Prediction:", decoded_preds[0])
    print(Fore.WHITE+"==========================================================================================================")
    