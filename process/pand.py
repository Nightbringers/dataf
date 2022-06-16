
import pandas as pd
import numpy as np
import tensorflow as tf
if __name__ == '__main__':
    # df = pd.read_csv('../data/News2022_train.tsv',sep='\t')
    # print(df)
    # pd.DataFrame()
    df2 = pd.read_csv('../data/train.tsv',sep='\t',header=None)
    # print(df2)
    query = df2[0]
    passage = df2[1]
    query = np.array(query)
    passage = np.array(passage)
    all_query = tf.data.Dataset.from_tensor_slices(query)
    all_pos = tf.data.Dataset.from_tensor_slices(passage)
    dataset = tf.data.Dataset.zip((all_query, all_pos))
    # dataset = dataset.shuffle(5000)
    batched_dataset = dataset.batch(5)
    for e in batched_dataset.take(1):
        print(e)

    # with open('../data/News2022_doc.tsv', 'r') as f:
    #     a = f.readlines()
    # with open('../data/News2022_train.tsv', 'r') as f:
    #     b = f.readlines()
    # print(len(a))
    # print(len(b))
    # a1 = a[1:]
    # b1 = b[1:]
    # # for i in range(len(a1)):
    # di = {}
    # ls2 = []
    # for e in a1:
    #     tmp = e.split('\t')
    #     # di[tmp[0]] = tmp[1].replace('\n','')
    #     di[tmp[0]] = tmp[1]
    # for e in b1:
    #     ls = []
    #     tmp = e.split('\t')
    #     ls.append(tmp[2].replace('\n',''))
    #     ls.append('_')
    #     ls.append(di[tmp[1]])
    #     s = '\t\t'.join(ls)
    #     ls2.append(s)
    # # print(ls2)
    # with open('../data/train.tsv', 'w') as f:
    #     f.writelines(ls2)
    #     # f.write("\n".join(ls2))
    # print('done')
    #     # di[tmp[0]] = tmp[1]
    #     # print(tmp)
    #
    # s = pd.read_csv('../data/News2022_doc.tsv',sep='\t')
    # print(s)
