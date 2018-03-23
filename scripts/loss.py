# Building a ranking loss function
import random
import json
import math
import numpy as np
import tensorflow as tf

from constants import HUMAN_ANNOTATION_PATH
# HUMAN_ANNOTATION_PATH = './human36m_corrected.json'
BATCH_SIZE = 64
BATCH_NUM_PAIRS = 16

def loss_numpy(pred, r):
    # print pred
    squared_diff = np.square(pred[:,:,1] - pred[:,:,0])
    # print squared_diff
    loss_partial = np.log(1 + np.exp(r * (pred[:,:,1] - pred[:,:,0])))
    goods = (r == 0)
    loss_partial[goods] = squared_diff[goods]
    # print loss_partial
    loss = np.sum(loss_partial)

    return loss

def loss_tf(pred, r):
    # pred = tf.Print(pred,[pred],message="pred ",summarize=128)

    squared_diff = tf.square(pred[:,:,1] - pred[:,:,0])
    # squared_diff = tf.Print(squared_diff,[squared_diff],message="squared_diff ",summarize=128)
    # squared_diff = tf.Print(squared_diff,[squared_diff.shape],message="squared_diff shape ",summarize=128)

    loss_temp = tf.log(1+tf.exp(r * (pred[:,:,1] - pred[:,:,0])))
    # loss_temp = tf.Print(loss_temp,[loss_temp.shape],message="loss_temp shape ",summarize=128)

    # loss_partial = tf.reshape(loss_temp,[-1,1])
    loss_partial = loss_temp
    # loss_partial = tf.Print(loss_partial,[loss_partial],message="loss_partial ", summarize=128)
    # loss_partial = tf.Print(loss_partial,[loss_partial.get_shape()],message="loss_partial shape ",summarize=128)

    good = tf.cast(tf.where(tf.equal(r, 0.)),tf.int32)
    # good = tf.Print(good,[good],message="good ", summarize=128)
    # good = tf.Print(good,[good.shape],message="goodshape ", summarize=128)
    gathered_good = tf.reshape(tf.gather_nd(squared_diff,good),[-1])
    # gathered_good = tf.Print(gathered_good,[gathered_good],message="gathered_good ", summarize=128)
    updates_gathered_good = tf.scatter_nd(good, gathered_good,  tf.shape(squared_diff))
    # updates_gathered_good = tf.Print(updates_gathered_good,[updates_gathered_good],message="updates_gathered_good ", summarize=128)
    # updates_gathered_good = tf.Print(updates_gathered_good,[updates_gathered_good.shape],message="updates_gathered_good.shape ", summarize=128)

    not_good = tf.cast(tf.where(tf.not_equal(r, 0.)),tf.int32)
    # not_good = tf.Print(not_good,[not_good],message="not_good ", summarize=128)
    # not_good = tf.Print(not_good,[not_good.get_shape()],message="not_good shape ", summarize=128)
    
    gathered_not_good = tf.reshape(tf.gather_nd(loss_partial,not_good),[-1])
    # gathered_not_good = tf.Print(gathered_not_good,[gathered_not_good],message="gathered_not_good ", summarize=128)
    
    updates_gathered_not_good = tf.scatter_nd(not_good, gathered_not_good,  tf.shape(squared_diff))
    # updates_gathered_not_good = tf.Print(updates_gathered_not_good,[updates_gathered_not_good],message="updates_gathered_not_good ", summarize=128)

    loss = tf.reduce_sum(updates_gathered_not_good + updates_gathered_good)
    # loss = tf.Print(loss,[loss],message="loss ",summarize=128)
    # loss = gathered_not_good
    return loss

def preloss_tf(y, dec_out, ps=None):
    # dec_out is truth
    # y is the pred
    if ps is not None:
        # ps = tf.Print(ps,[ps],message="ps ",summarize=128)
        # ps = tf.Print(ps,[ps.shape],message="ps.shape ",summarize=128)
        # dec_out = tf.Print(dec_out,[dec_out],message="dec_out ",summarize=128)
        # dec_out = tf.Print(dec_out,[dec_out.shape],message="dec_out.shape ",summarize=128)
        dec_out = tf.gather(dec_out, ps, axis=1)
        dec_out = tf.gather_nd(dec_out, [(_i, _i) for _i in range(64)])
        # dec_out = dec_out[0]
        y = tf.gather(y, ps, axis=1)
        y = tf.gather_nd(y, [(_i, _i) for _i in range(64)])
        # y = y[0]
    # dec_out = tf.Print(dec_out,[dec_out],message="dec_out ",summarize=128)
    # dec_out = tf.Print(dec_out,[dec_out[0]],message="dec_out[0] ",summarize=128)
    # dec_out = tf.Print(dec_out,[dec_out[1]],message="dec_out[1] ",summarize=128)
    # dec_out = tf.Print(dec_out,[dec_out.shape],message="dec_out.shape ",summarize=128)

    r = tf.sign(dec_out[:,:,0] - dec_out[:,:,1])

    # ## MAKE SOME FAKE DATA WHERE r = 0
    # indices = [[0, 0], [0, 1]]  # A list of coordinates to update.
    # values = [1.0, 1.0]  # A list of values corresponding to the respective
    # shape = [64, 16]  # The shape of the corresponding dense tensor, same as `c`.
    # delta = tf.cast(tf.SparseTensor(indices, values, shape), tf.float64)
    # r = r + tf.sparse_tensor_to_dense(delta)

    # r = tf.Print(r,[r],message="r ",summarize=32)
    return loss_tf(y, r)

def main():

    sess = tf.InteractiveSession()

    with open(HUMAN_ANNOTATION_PATH) as f:
        _human_dataset = json.load(f)

    num_pts = len(_human_dataset['annotations'][0]['kpts_3d']) / 3
    pairs = np.asarray([(i, j) for i in range(num_pts) for j in range(num_pts) if i < j])

    np_truth_all = [_human_dataset['annotations'][i]['kpts_3d'][1::3] for i in range(BATCH_SIZE)]
    np_truth_all = np.asarray(np_truth_all)
    np_truth_all = (np_truth_all - np.mean(np_truth_all)) / np.std(np_truth_all)
    # print "np_truth_all.shape ", np_truth_all.shape

    pair_idxs = np.random.choice(len(pairs), size=(BATCH_SIZE, BATCH_NUM_PAIRS))
    pair_idxs = np.asarray([np.random.choice(len(pairs), BATCH_NUM_PAIRS, replace=False) for _b in range(BATCH_SIZE)])
    np_ps = np.take(pairs, pair_idxs, axis=0)
    # print np_ps
    # print np_ps.shape

    np_truth = np.take(np_truth_all, np_ps, axis=1)
    np_truth = np.asarray([np_truth[_i][_i] for _i in range(BATCH_SIZE)])
    # print "np_truth_all", np_truth_all
    # print "np_truth", np_truth
    # print np_truth.shape

    np_r = np.sign(np_truth[:,:,0] - np_truth[:,:,1], dtype=float)
    # np_r[np.random.randint(64, size=np.random.randint(10))] = 0
    # np_r[:][np.random.randint(BATCH_NUM_PAIRS, size=np.random.randint(10))] = 0
    # np_r[0][0] += 1
    # np_r[0][1] += 1
    # print np_r

    # numpy loss
    np_loss = loss_numpy(np_truth, np_r)

    # convert numpy arrays to tf tensors
    tf_truth_all = tf.convert_to_tensor(np_truth_all)
    tf_truth = tf.convert_to_tensor(np_truth)
    # tf_truth = tf.Print(tf_truth,[tf_truth],message="tf_truth ",summarize=128)
    tf_r = tf.convert_to_tensor(np_r)
    # tf_r = tf.Print(tf_r,[tf_r],message="tf_r ",summarize=128)
    tf_ps = tf.convert_to_tensor(np_ps)
    # tf_ps = tf.Print(tf_ps,[tf_ps],message="tf_ps ",summarize=128)
    tf_pairs = tf.convert_to_tensor(pairs)

    # tf loss
    # tf_loss = preloss_tf(tf_truth_all, tf_truth_all, ps=tf_ps)
    loss = preloss_tf(tf_truth_all, tf_truth_all, ps=tf_ps)
    # tf_loss = loss_tf(tf_truth, tf_r)
    tf_loss = sess.run([loss])

    print (np_loss, tf_loss)

    print ("Done")

if __name__ == '__main__':
    main()
