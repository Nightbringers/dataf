import tensorflow as tf

def create_ds(epoch,batch):
        path = '../data/News2022_doc.tsv'
        data = tf.data.experimental.make_csv_dataset(
                path,
                batch_size=batch,  # Artificially small to make examples easier to show.
                num_epochs=epoch,
                field_delim='\t',
                # header=False,
                # column_names=['l0','l1','l2','l3','l4','l5'],
                shuffle=False,
                # shuffle_buffer_size = 500000,
                ignore_errors=True)

        return data

if __name__ == '__main__':


    # a = create_ds(1,5)
    with open('../data/News2022_doc.tsv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(len(lines))
    print(1)
    # for e in a.take(1):
    #         print(e)




