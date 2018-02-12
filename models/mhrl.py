################################################################################
# CS81 Depth Research
# Robert Eng
#
# It is named mhrl by taking the first letter in each of the four authors'
# names: Martinez, Hossain, Romero, Little.
#
# Goal: Implement the 2d to 3d pose estimation model in this paper and reproduce
# their results.
# https://arxiv.org/abs/1705.03098
#
# The original code is at: https://github.com/una-dinosauria/3d-pose-baseline
# My code pulls from their code at times when what they do is trivial.
#
# To enable anaconda, `source activate tensorflow`
################################################################################

import os
import sys
import json
import numpy as np

# from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout, Add, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/Robert/Documents/Caltech/CS81_Depth_Research/scripts/')
from postprocess_original_utils import correct_lean


################################################################################
# Configuration
################################################################################

HUMAN_ANNOTATION_DIR  = '/Users/Robert/Documents/Caltech/CS81_Depth_Research/' \
                        'datasets/human36m_annotations'
HUMAN_ANNOTATION_FILE = 'human36m_train.json'
HUMAN_ANNOTATION_PATH = os.path.join(HUMAN_ANNOTATION_DIR, HUMAN_ANNOTATION_FILE)

CHECKPOINTS_DIR = '../checkpoints'

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS  = [9, 11]

NUM_EPOCHS = 200
LIMIT = None

# NUM_EPOCHS = 10
# LIMIT = 1000

################################################################################
# Preprocessing
################################################################################


def get_human_data(limit=None):
    with open(HUMAN_ANNOTATION_PATH) as f:
        _human_dataset = json.load(f)
        correct_lean(_human_dataset)
    if limit is not None:
        train = [h['kpts_2d'] for h in _human_dataset['annotations'] if h['s_id'] in TRAIN_SUBJECTS][:LIMIT]
        test  = [h['kpts_2d'] for h in _human_dataset['annotations'] if h['s_id'] in TEST_SUBJECTS][:LIMIT]
        trainlabels = [h['kpts_3d'] for h in _human_dataset['annotations'] if h['s_id'] in TRAIN_SUBJECTS][:LIMIT]
        testlabels  = [h['kpts_3d'] for h in _human_dataset['annotations'] if h['s_id'] in TEST_SUBJECTS][:LIMIT]
    else:
        train = [h['kpts_2d'] for h in _human_dataset['annotations'] if h['s_id'] in TRAIN_SUBJECTS]
        test  = [h['kpts_2d'] for h in _human_dataset['annotations'] if h['s_id'] in TEST_SUBJECTS]
        trainlabels = [h['kpts_3d'] for h in _human_dataset['annotations'] if h['s_id'] in TRAIN_SUBJECTS]
        testlabels  = [h['kpts_3d'] for h in _human_dataset['annotations'] if h['s_id'] in TEST_SUBJECTS]

    keypoints = _human_dataset['pose'][0]['keypoints']
    assert len(train) == len(trainlabels)
    assert len(test) == len(testlabels)
    return train, test, trainlabels, testlabels, keypoints


def normalize_data(train, test, trainlabels, testlabels):
    train_mean = np.mean(train, axis=0)
    train_std  = np.std(train, axis=0)
    trainlabels_mean = np.mean(trainlabels, axis=0)
    trainlabels_std  = np.std(trainlabels, axis=0)

    train_normed = (train - train_mean) / train_std
    test_normed  = (test - train_mean) / train_std

    trainlabels_normed = (trainlabels - trainlabels_mean) / trainlabels_std
    testlabels_normed  = (testlabels - trainlabels_mean) / trainlabels_std

    assert (train[0][0] - train_mean[0]) / train_std[0] == train_normed[0][0]
    assert (train[1][1] - train_mean[1]) / train_std[1] == train_normed[1][1]
    assert (test[0][0] - train_mean[0]) / train_std[0] == test_normed[0][0]
    assert (test[1][1] - train_mean[1]) / train_std[1] == test_normed[1][1]
    assert (trainlabels[0][0] - trainlabels_mean[0]) / trainlabels_std[0] == trainlabels_normed[0][0]
    assert (trainlabels[1][1] - trainlabels_mean[1]) / trainlabels_std[1] == trainlabels_normed[1][1]
    assert (testlabels[0][0] - trainlabels_mean[0]) / trainlabels_std[0] == testlabels_normed[0][0]
    assert (testlabels[1][1] - trainlabels_mean[1]) / trainlabels_std[1] == testlabels_normed[1][1]

    return train_normed, test_normed, trainlabels_normed, testlabels_normed





################################################################################
# Models and Training
################################################################################


def residual_block(x):
    # core residual unit
    residual = x
    # Linear
    x = Dense(1024, kernel_initializer='he_normal')(x)
    # Batch Norm
    x = BatchNormalization()(x)
    # RELU
    x = Activation('relu')(x)
    # Dropout
    x = Dropout(0.5)(x)
    # Linear
    x = Dense(1024, kernel_initializer='he_normal')(x)
    # Batch Norm
    x = BatchNormalization()(x)
    # RELU
    x = Activation('relu')(x)
    # Dropout
    x = Dropout(0.5)(x)
    # Add residual back in
    x = Add()([residual, x])
    return x

def find_latest_epoch():
    '''Iterate through checkpoints directory looking for latest epoch'''
    latest_epoch = -1
    latest_checkpoint_filepath = None
    for file in os.listdir(CHECKPOINTS_DIR):
        if file.startswith("mhrl-"):
            epoch = parse_epoch_from_checkpoint_filepath(file)
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint_filepath = file
    return latest_epoch, latest_checkpoint_filepath

def parse_epoch_from_checkpoint_filepath(checkpoint_filepath):
    return int(checkpoint_filepath.split('-')[1])

def create_mhrl_model(num_kpts):
    inputs = Input(shape=(2 * num_kpts,))

    x = Dense(1024, kernel_initializer='he_normal')(inputs)

    x = residual_block(x)
    x = residual_block(x)
    x = Dense(3 * num_kpts, kernel_initializer='he_normal')(x)
    
    model = Model(inputs=[inputs], outputs=[x])

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse'])
    return model

def train_model(train, test, trainlabels, testlabels, model, num_epochs, use_latest_checkpoint=False):
    filepath="../checkpoints/mhrl-{epoch:03d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, period=10, mode='min')
    callbacks_list = [checkpoint]

    # If using a checkpoint, load the checkpoint and epoch
    initial_epoch = 0
    if use_latest_checkpoint:
        latest_epoch, checkpoint_filepath = find_latest_epoch()
        if checkpoint_filepath is not None:
            model.load_weights(os.path.join(CHECKPOINTS_DIR, checkpoint_filepath))
            initial_epoch = latest_epoch
            print "Using checkpoint {} at epoch {}".format(checkpoint_filepath, latest_epoch)

    history = model.fit(train, trainlabels,
                        epochs=num_epochs,
                        batch_size=64,
                        validation_data=[test, testlabels],
                        callbacks=callbacks_list,
                        initial_epoch=initial_epoch)

    scores = model.evaluate(test, testlabels, verbose=0)
    print("Error: %.2f%%" % (100-scores[1]*100))
    return history
    


################################################################################
# Postprocessing
################################################################################

def plot_training(history):
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model mean_squared_error')
    plt.ylabel('mean_squared_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

################################################################################
# Main
################################################################################

# train, test, trainlabels, testlabels, keypoints = get_human_data()
train, test, trainlabels, testlabels, keypoints = get_human_data(limit=LIMIT)
train, test, trainlabels, testlabels = normalize_data(train, test, trainlabels, testlabels)
# TODO: Normalize based on camera view?

model = create_mhrl_model(len(keypoints))
history = train_model(train, test, trainlabels, testlabels, model, NUM_EPOCHS,
                      use_latest_checkpoint=True)

plot_training(history)



def main():
    print "main"
    # train, test, trainlabels, testlabels = get_bird_datasets(DATA_SET_TYPE)
    # ResNet(train, test, trainlabels, testlabels)
    # # analyze_performance(train, test, trainlabels, testlabels)
    
    # # find_differences()

    # train_outputs, test_outputs, trainlabels, testlabels = get_Uberklass_features()
    # # train_Uberklass(train_outputs, test_outputs, trainlabels, testlabels)
    # analyze_Uberklass(testlabels)


if __name__ == "__main__":
    main()