
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
if __name__ == '__main__':
    path = '../data/pretrain_model/roberta'
    tokenizer = BertTokenizer.from_pretrained("roberta-base")
    model = TFBertModel.from_pretrained("roberta-base")

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)