# Building a ranking loss function
import random
import json
import math
import numpy as np
import tensorflow as tf

from constants import HUMAN_ANNOTATION_PATH
BATCH_NUM_PAIRS = 64

def loss_numpy(pred, ps, r):
    pred = np.take(pred, ps)

    #Just the r != 0. Probably faster.
    # return np.sum(np.log(1 + np.exp(r * np.subtract(pred[:,1], pred[:,0]))))

    # np piecewise function
    # return np.sum(np.piecewise(r, [r == 0, r != 0], [lambda r: np.square(pred[:,1] - pred[:,0]), lambda r: np.log(1 + np.exp(r * np.subtract(pred[:,1], pred[:,0])))]))    

    # Writeover r = 0 case
    result = np.log(1 + np.exp(r * (pred[:,1] - pred[:,0])))
    goods = (r == 0)
    diffs = np.square(pred[:,1] - pred[:,0])
    result[goods] = diffs[goods]
    return np.sum(result)

def loss_tf(pred, ps, r):

    diffs = tf.square(tf.subtract(pred[:,1], pred[:,0]))
    # Writeover r = 0 case
    result = tf.log(tf.add(tf.cast(1.,tf.float64), tf.exp(tf.multiply(r, diffs))))
    goods = tf.cast(tf.where(tf.equal(r, 0.))[1],tf.int32)
    goods = tf.Print(goods,[goods],summarize=128)
    
    
    updates = tf.scatter_nd(goods,diffs,tf.shape(result))

    mask = tf.multiply(result, tf.cast(tf.not_equal(r, 0.),tf.float64))
    result = updates + mask

    # result = tf.scatter_nd(result, goods, diffs)

    return tf.reduce_sum(result)

def main():
    with open(HUMAN_ANNOTATION_PATH) as f:
        _human_dataset = json.load(f)

    num_pts = len(_human_dataset['annotations'][0]['kpts_3d']) / 3

    pairs = np.asarray([(i, j) for i in range(num_pts) for j in range(num_pts) if i != j])

    np_truth = _human_dataset['annotations'][0]['kpts_3d'][1::3]
    np_truth = np.asarray(np_truth)
    np_truth = (np_truth - np.mean(np_truth)) / np.std(np_truth)

    pair_idxs = np.random.choice(len(pairs), BATCH_NUM_PAIRS)
    np_ps = np.take(pairs, pair_idxs, axis=0)

    np_truth = np.take(np_truth, np_ps)
    r = np.sign(np_truth[:,0] - np_truth[:,1], dtype=float)
    print np_truth
    print r

    print loss_numpy(np_truth, np_ps, r)

    # sess = tf.InteractiveSession()
    # truth = tf.placeholder("float64", [64, 2])
    # ps = tf.placeholder("float64", [64, 2])
    with tf.Session() as sess:
        truth = tf.convert_to_tensor(np_truth)
        
        ps = tf.convert_to_tensor(np_ps)
        sub = tf.subtract(tf.slice(truth, [0,0],[64, 1]), tf.slice(truth, [0,1],[64, 1]))
        r = tf.sign(sub)

        # truth = tf.Print(truth, [truth], message="truth ", summarize=128)
        # ps = tf.Print(ps, [ps], message="ps ", summarize=128)
        # r = tf.Print(r, [r], message="r ", summarize=128)

        # tf_loss = loss_tf(truth, ps, r)

        diffs = tf.square(sub)
        diffs = tf.Print(diffs,[diffs],message="diffs",summarize=128)
        print("diffs")
        print(diffs.shape)
        
        result = tf.log(tf.add(tf.cast(1.,tf.float64), tf.exp(tf.multiply(r, diffs))))
        result = tf.Print(result,[result],message="result",summarize=128)
        print("results")
        print(result.shape)

        goods = tf.reshape(tf.where(tf.not_equal(r, 0.)),[-1,2])
        goods = tf.Print(goods,[goods],message="goods",summarize=128)
        print("goods")
        print(goods.shape)
        
        # updates = tf.scatter_nd(goods,diffs,result.shape)
        # updates = tf.Print(updates,[updates],message="updates",summarize=128)
        # print("updates")
        # print(updates.shape)

        # mask = tf.multiply(result, tf.cast(tf.not_equal(r, 0.),tf.float64))
        # result = updates + mask

        # result = tf.scatter_nd(result, goods, diffs)

        # tf_loss = tf.reduce_sum(result)

        sess.run([diffs, result, goods])
        # sess.run(result,feed_dict={truth: np_truth, ps: np_ps})


if __name__ == '__main__':
    main()
