
import tensorflow as tf
from model import search
from  reader import create_ds
import pandas as pd
import numpy as np
if __name__ == '__main__':
    df2 = pd.read_csv('../data/train.tsv', sep='\t', header=None)
    # print(df2)
    query = df2[0]
    passage = df2[1]
    query = np.array(query)
    passage = np.array(passage)
    all_query = tf.data.Dataset.from_tensor_slices(query)
    model = search()