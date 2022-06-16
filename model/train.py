
import tensorflow as tf
import collections
import logging
import os
from pprint import pprint
import numpy as np
import tensorflow as tf
import pandas as pd
from model import search

from transformers import AutoTokenizer, BertTokenizer,TFBertModel

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


@tf.function
def cos_sim(input, label):
    x = input * label
    a = tf.reduce_sum(x, axis=-1)
    input_l = input * input
    b = tf.reduce_sum(input_l, axis=-1)
    b = tf.sqrt(b)
    label_l = label * label
    c = tf.reduce_sum(label_l, axis=-1)
    c = tf.sqrt(c)
    fin = a / (b * c)
    fin = tf.expand_dims(fin, axis=-1)
    # tf.reshape(fin,[fin.shape[0],1])
    return fin

loss_cr = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

with strategy.scope():
    def loss_my_c(pos, tar,neg,pos_emb_2,squeue):
        # a = tf.transpose(bat)
        a = tf.matmul(pos, tar, transpose_b=True)

        pos_norm = tf.norm(pos, axis=-1, keepdims=True)

        tar_norm = tf.norm(tar, axis=-1, keepdims=True)
        d = tf.matmul(pos_norm, tar_norm, transpose_b=True)
        cos_mat = a / d
        cos_mat = cos_mat / 0.05
        a_squeue = tf.matmul(pos, squeue, transpose_b=True)
        squeue_norm = tf.norm(squeue, axis=-1, keepdims=True)
        squeue_mat_t = tf.matmul(pos_norm, squeue_norm, transpose_b=True)
        squeue_mat = a_squeue / squeue_mat_t
        squeue_mat = squeue_mat / 0.05

        squeue_sim = tf.exp(squeue_mat)
        # print(b1.shape)

        squeue_sum = tf.reduce_sum(squeue_sim, axis=-1, keepdims=True)

        neg_norm = tf.norm(neg, axis=-1, keepdims=True)

        neg_a = tf.matmul(pos, neg, transpose_b=True)

        neg_ma = tf.matmul(pos_norm, neg_norm, transpose_b=True)

        a_hardneg = neg_a / neg_ma
        a_hardneg =  a_hardneg/ 0.05

        top = cos_sim(pos, pos_emb_2) / 0.05
        # top_neg2 = cos_sim(pos, neg_2) / 0.05

        p = tf.exp(top)
        # p_neg2 = tf.exp(top_neg2)
        n = tf.exp(cos_mat)
        hard_neg_sim = tf.exp(a_hardneg)
        hard_neg_sum = tf.reduce_sum(hard_neg_sim, axis=-1, keepdims=True)

        n_sum = tf.reduce_sum(n, axis=-1, keepdims=True)
        x = p / (n_sum  + hard_neg_sum)
        loss = tf.math.log(x)
        loss = -loss
        loss = tf.reduce_sum(loss)
        # b = tf.eye(a.shape[0])
        # loss = loss_cr(b, cos_mat)
        return loss


with strategy.scope():
    def loss_cont(pos, tar, squeue):
        # a = tf.matmul(pos, tar, transpose_b=True)
        pos_norm = tf.norm(pos, axis=-1, keepdims=True)
        # tar_norm = tf.norm(tar, axis=-1, keepdims=True)
        # d = tf.matmul(pos_norm, tar_norm, transpose_b=True)
        # cos_mat = a / d
        # a_hardneg = cos_sim(pos, neg) / 0.04
        a_squeue = tf.matmul(pos, squeue, transpose_b=True)
        squeue_norm = tf.norm(squeue, axis=-1, keepdims=True)
        squeue_mat_t = tf.matmul(pos_norm, squeue_norm, transpose_b=True)
        squeue_mat = a_squeue / squeue_mat_t
        squeue_mat = squeue_mat /0.05
        # cos_mat = cos_mat / 0.05

        top = cos_sim(pos, tar) /0.05

        p = tf.exp(top)

        # n = tf.exp(cos_mat)

        squeue_sim = tf.exp(squeue_mat)
        # print(b1.shape)

        squeue_sum = tf.reduce_sum(squeue_sim, axis=-1, keepdims=True)


        x = p / (squeue_sum)
        loss = tf.math.log(x)
        loss = -loss
        loss = tf.reduce_sum(loss)
        # b = tf.eye(a.shape[0])
        # loss =loss_cr(b,cos_mat)
        return loss
with strategy.scope():
    def loss_cont_d(pos, tar,squeue):
        # a = tf.matmul(pos, tar, transpose_b=True)
        pos_norm = tf.norm(pos, axis=-1, keepdims=True)
        # tar_norm = tf.norm(tar, axis=-1, keepdims=True)
        # d = tf.matmul(pos_norm, tar_norm, transpose_b=True)
        # cos_mat = a / d
        # a_hardneg = cos_sim(pos, neg) / 0.04
        a_squeue = tf.matmul(pos, squeue, transpose_b=True)
        squeue_norm = tf.norm(squeue, axis=-1, keepdims=True)
        squeue_mat_t = tf.matmul(pos_norm, squeue_norm, transpose_b=True)
        squeue_mat = a_squeue / squeue_mat_t
        squeue_mat = squeue_mat /0.05
        # cos_mat = cos_mat / 0.05

        top = cos_sim(pos, tar) / 0.05
        # top_neg = cos_sim(neg, tar) /0.05
        p = tf.exp(top)
        # p_neg = tf.exp(top_neg)

        # n = tf.exp(cos_mat)
        squeue_sim = tf.exp(squeue_mat)
        # print(b1.shape)

        squeue_sum = tf.reduce_sum(squeue_sim, axis=-1, keepdims=True)

        x = p / (squeue_sum)
        loss = tf.math.log(x)
        loss = -loss
        loss = tf.reduce_sum(loss)
        # b = tf.eye(a.shape[0])
        # loss =loss_cr(b,cos_mat)
        return loss


if __name__ == '__main__':
    df2 = pd.read_csv('../data/train.tsv', sep='\t', header=None)
    # print(df2)
    query = df2[0]
    passage = df2[1]
    query = np.array(query)
    passage = np.array(passage)
    all_query = tf.data.Dataset.from_tensor_slices(query)
    all_pos = tf.data.Dataset.from_tensor_slices(passage)
    dataset = tf.data.Dataset.zip((all_query, all_pos))
    # dataset = dataset.shuffle(5000)
    ds = dataset.batch(4800)

    # for e in ds.take(1):
    #     print(e)
    with strategy.scope():
        q_model = search()
        p_model = search()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
        checkpoint = tf.train.Checkpoint(model=q_model, step=tf.Variable(0))
        checkpoint_manager = tf.train.CheckpointManager(checkpoint,'../save/q1', 3)
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        if checkpoint_manager.latest_checkpoint:
            logging.info("Restored from {}".format(checkpoint_manager.latest_checkpoint))
        else:
            logging.info("Initializing from scratch.")

        checkpoint_p = tf.train.Checkpoint(model=p_model, step=tf.Variable(0))
        checkpoint_manager_p = tf.train.CheckpointManager(checkpoint_p,'../save/p1', 3)
        checkpoint_p.restore(checkpoint_manager_p.latest_checkpoint)
        if checkpoint_manager_p.latest_checkpoint:
            logging.info("Restored from {}".format(checkpoint_manager_p.latest_checkpoint))
        else:
            logging.info("Initializing from scratch.")


    @tf.function
    def train_step_q2d(inputs, squeue):
        query, pos = inputs
        pos_emb_2 = p_model(pos, training=False)
        with tf.GradientTape() as tape:
            query_emb = q_model(query, training=True)
            # query_emb = q_model(query, training=True)
            loss = loss_cont(query_emb, pos_emb_2, squeue)
        grads = tape.gradient(loss, q_model.trainable_weights)
        # grads2 = tape.gradient(loss, p_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, q_model.trainable_weights))
        # optimizer.apply_gradients(zip(grads2, p_model.trainable_weights))
        return loss


    @tf.function
    def train_step_d2q(inputs, squeue):
        query, pos = inputs
        query_emb_2 = q_model(query, training=False)
        with tf.GradientTape() as tape:
            pos_emb = p_model(pos, training=True)
            # neg_emb2 = p_model(neg, training=True)
            # neg_emb = p_model(neg)
            # query_emb = q_model(query, training=True)
            loss = loss_cont_d(pos_emb, query_emb_2, squeue)
        grads = tape.gradient(loss, p_model.trainable_weights)
        # grads2 = tape.gradient(loss, p_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, p_model.trainable_weights))
        # optimizer.apply_gradients(zip(grads2, p_model.trainable_weights))
        return loss

    def train_s(inputs):
        query, pos = inputs
        pos_emb = p_model(pos, training=False)
        query_emb = q_model(query, training=False)
        return pos_emb,query_emb

    @tf.function
    def distributed_train_step(dist_inputs,squeue):
        per_replica_losses = strategy.run(train_step_q2d, args=(dist_inputs,squeue))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    @tf.function
    def distributed_train_step2(dist_inputs,squeue):
        per_replica_losses = strategy.run(train_step_d2q, args=(dist_inputs, squeue))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    @tf.function
    def return_emb_dist(dist_inputs):
        pos_emb, query_emb = strategy.run(train_s, args=(dist_inputs,))
        # fin = strategy.gather(pass_emb, axis=0)
        return strategy.gather(pos_emb, axis=0), strategy.gather(query_emb, axis=0)

    flag = 0

    for e in ds:
        squeue = []
        batched_dataset = e.batch(60)
        dist_dataset = strategy.experimental_distribute_dataset(batched_dataset)
        if flag == 0:
            for elment in dist_dataset:
                pos_emb, query_emb = return_emb_dist(elment)
                squeue.append(pos_emb)

            for elment in dist_dataset:
                checkpoint.step.assign_add(1)
                step = checkpoint.step.numpy()
                # pos_emb, query_emb, neg_emb = return_emb_dist(elment)
                # squeue.append(query_emb)
                # squeue.append(neg_emb)
                k = squeue[:]
                # k = np.array(squeue)
                # tem_squeue = np.concatenate(k,axis=0)
                tem_squeue = tf.concat(k, axis=0)

                loss1 = distributed_train_step(elment, tem_squeue)

                # pos_emb, query_emb, neg_emb = return_emb_dist(elment)
                # loss2 = distributed_train_step2(elment,query_emb,neg_emb)
                if int(step) % 20 == 0:
                    print(tem_squeue.shape)
                    print("loss1 {:1.2f}".format(loss1.numpy()))
                    # print("loss2 {:1.2f}".format(loss2.numpy()))
                    # print("loss3 {:1.2f}".format(loss3.numpy()))

                if int(step) % 300 == 0:
                    checkpoint_manager.save(checkpoint_number=step)
                    checkpoint_manager_p.save(checkpoint_number=step)
                    # best_m1 = m1
                    print("Saved checkpoint")
                    print(step)
                    print("loss1 {:1.2f}".format(loss1.numpy()))
                    # print("loss2 {:1.2f}".format(loss2.numpy()))
                    # print("loss3 {:1.2f}".format(loss3.numpy()))

                    # print("loss {:1.2f}".format(loss_03.numpy()))
            checkpoint_manager.save(checkpoint_number=step)
            checkpoint_manager_p.save(checkpoint_number=step)
            flag = 1
        else:
            squeue = []
            for elment in dist_dataset:
                pos_emb, query_emb, neg_emb = return_emb_dist(elment)
                squeue.append(query_emb)
            for elment in dist_dataset:
                checkpoint.step.assign_add(1)
                step = checkpoint.step.numpy()
                k = squeue[:]
                tem_squeue = tf.concat(k, axis=0)
                loss2 = distributed_train_step2(elment, tem_squeue)

                # pos_emb, query_emb, neg_emb = return_emb_dist(elment)
                # loss2 = distributed_train_step2(elment,query_emb,neg_emb)
                if int(step) % 20 == 0:
                    print(tem_squeue.shape)
                    print("loss2 {:1.2f}".format(loss2.numpy()))
                    # print("loss2 {:1.2f}".format(loss2.numpy()))
                    # print("loss3 {:1.2f}".format(loss3.numpy()))

                if int(step) % 300 == 0:
                    checkpoint_manager.save(checkpoint_number=step)
                    checkpoint_manager_p.save(checkpoint_number=step)
                    # best_m1 = m1
                    print("Saved checkpoint")
                    print(step)
                    print("loss2 {:1.2f}".format(loss2.numpy()))
                    # print("loss2 {:1.2f}".format(loss2.numpy()))
                    # print("loss3 {:1.2f}".format(loss3.numpy()))

                    # print("loss {:1.2f}".format(loss_03.numpy()))
            checkpoint_manager.save(checkpoint_number=step)
            checkpoint_manager_p.save(checkpoint_number=step)
            flag = 0
    print('done')






