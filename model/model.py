#Model Import
from transformers import T5ForConditionalGeneration, T5Tokenizer
from config.config import cfg
model = T5ForConditionalGeneration.from_pretrained(cfg['model']['name'])
tokenizer = T5Tokenizer.from_pretrained(cfg['model']['name'])

