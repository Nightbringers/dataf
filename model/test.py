import tensorflow as tf
from reader import create_ds

def create_padding_mask(seq):
    seq = tf.cast(tf.math.not_equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq

if __name__ == '__main__':
    a = create_ds(1, 3000000)
    input_text_processor = tf.keras.layers.TextVectorization(max_tokens=100000)
    for e in a.take(1):
        print(111)
        input_text_processor.adapt(e['docid'])
    vob = input_text_processor.get_vocabulary()
    with open('../data/vocab_100000.txt', 'w') as f:
        f.writelines("\n".join(vob))
    print(len(vob))
    print(vob[:100])
    print(vob[-100:])