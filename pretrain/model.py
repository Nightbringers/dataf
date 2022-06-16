
from transformers import TFBertModel

from tensorflow.python.keras.models import Model

import tensorflow as tf

 # (batch_size, 1, 1, seq_len)

class search(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding = tf.keras.layers.Embedding(100000,768)

        self.bert = TFBertModel.from_pretrained('../data/roberta')


    def call(self,inputs,training,**kwargs):
        mask = self.create_padding_mask(inputs)
        token_emb = self.embedding(inputs)
        emb = self.bert(input_ids=None, attention_mask=mask, inputs_embeds=token_emb,
                     training=training)['last_hidden_state']
        # emb = self.bert(inputs,training=training)['last_hidden_state']
        att_mask = inputs['attention_mask']
        att_mask = tf.cast(att_mask, tf.float32)
        txt_len = tf.reduce_sum(att_mask, axis=-1, keepdims=True)
        att = tf.expand_dims(att_mask,-1)

        emb = emb*att
        final = tf.reduce_sum(emb,axis=1)/txt_len
        return  final,emb,mask

    def create_padding_mask(seq):
        seq = tf.cast(tf.math.not_equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq





