
from transformers import TFBertModel

from tensorflow.python.keras.models import Model

import tensorflow as tf

class search(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = TFBertModel.from_pretrained('../data/tf_mengzi_base')

    def call(self,inputs,training,**kwargs):
        emb = self.bert(inputs,training=training)['last_hidden_state']
        att_mask = inputs['attention_mask']
        att_mask = tf.cast(att_mask, tf.float32)
        txt_len = tf.reduce_sum(att_mask, axis=-1, keepdims=True)
        att = tf.expand_dims(att_mask,-1)

        emb = emb*att
        final = tf.reduce_sum(emb,axis=1)/txt_len
        return  final



