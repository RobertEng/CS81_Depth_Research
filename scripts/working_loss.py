# Building a ranking loss function
import random
import json
import math
import numpy as np
import tensorflow as tf

from constants import HUMAN_ANNOTATION_PATH
# HUMAN_ANNOTATION_PATH = './human36m_corrected.json'
BATCH_NUM_PAIRS = 64

def loss_numpy(pred, r):

    squared_diff = np.square(pred[:,1] - pred[:,0])
    loss_partial = np.log(1 + np.exp(r * (pred[:,1] - pred[:,0])))
    goods = (r == 0)
    loss_partial[goods] = squared_diff[goods]
    loss = np.sum(loss_partial)

    return loss

def loss_tf(pred, r):

    squared_diff = tf.square(pred[:,1] - pred[:,0])
    # squared_diff = tf.Print(squared_diff,[squared_diff],message="squared_diff ",summarize=128)

    loss_partial = tf.reshape(tf.log(1+tf.exp(r * (pred[:,1] - pred[:,0]))),[-1,1])
    # loss_partial = tf.Print(loss_partial,[loss_partial],message="loss_partial ", summarize=128)

    good = tf.cast(tf.where(tf.equal(r, 0.)),tf.int32)
    # good = tf.Print(good,[good],message="good ", summarize=128)
    gathered_good = tf.reshape(tf.gather(squared_diff,good),[-1])
    # gathered_good = tf.Print(gathered_good,[gathered_good],message="gathered_good ", summarize=128)
    updates_gathered_good = tf.scatter_nd(good, gathered_good,  tf.shape(squared_diff))
    # updates_gathered_good = tf.Print(updates_gathered_good,[updates_gathered_good],message="updates_gathered_good ", summarize=128)

    not_good = tf.cast(tf.where(tf.not_equal(r, 0.)),tf.int32)
    # not_good = tf.Print(not_good,[not_good],message="not_good ", summarize=128)
    gathered_not_good = tf.reshape(tf.gather(loss_partial,not_good),[-1])
    # gathered_not_good = tf.Print(gathered_not_good,[gathered_not_good],message="gathered_not_good ", summarize=128)
    updates_gathered_not_good = tf.scatter_nd(not_good, gathered_not_good,  tf.shape(squared_diff))
    # updates_gathered_not_good = tf.Print(updates_gathered_not_good,[updates_gathered_not_good],message="updates_gathered_not_good ", summarize=128)

    loss = tf.reduce_sum(updates_gathered_not_good + updates_gathered_good)
    # loss = tf.Print(loss,[loss],message="loss ",summarize=128)

    return loss

def main():

    sess = tf.InteractiveSession()

    with open(HUMAN_ANNOTATION_PATH) as f:
        _human_dataset = json.load(f)

    num_pts = len(_human_dataset['annotations'][0]['kpts_3d']) / 3
    pairs = np.asarray([(i, j) for i in range(num_pts) for j in range(num_pts) if i < j])
    np_truth = _human_dataset['annotations'][0]['kpts_3d'][1::3]
    np_truth = np.asarray(np_truth)
    np_truth = (np_truth - np.mean(np_truth)) / np.std(np_truth)
    pair_idxs = np.random.choice(len(pairs), BATCH_NUM_PAIRS)
    np_ps = np.take(pairs, pair_idxs, axis=0)
    np_truth = np.take(np_truth, np_ps)
    # print np_truth, "MUST CHECK THAT SAME PAIR IS NOT PICKED TWICE, i.e. [1 11] [11 1]"

    np_r = np.sign(np_truth[:,0] - np_truth[:,1], dtype=float)
    np_r[np.random.randint(64, size=np.random.randint(10))] = 0

    # numpy loss
    np_loss = loss_numpy(np_truth, np_r)

    # convert numpy arrays to tf tensors
    tf_truth = tf.convert_to_tensor(np_truth)
    # tf_truth = tf.Print(tf_truth,[tf_truth],message="tf_truth ",summarize=128)
    tf_r = tf.convert_to_tensor(np_r)
    # tf_r = tf.Print(tf_r,[tf_r],message="tf_r ",summarize=128)
    tf_ps = tf.convert_to_tensor(np_ps)
    # tf_ps = tf.Print(tf_ps,[tf_ps],message="tf_ps ",summarize=128)

    # tf loss
    tf_loss = loss_tf(tf_truth, tf_r)
    tf_loss = sess.run([tf_loss])

    print np_loss, tf_loss

    print "Done"

if __name__ == '__main__':
    main()
