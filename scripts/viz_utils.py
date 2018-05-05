import os
import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle

from constants import (HUMAN_IMAGES_DIR, I_BASELINE, J_BASELINE, LR_BASELINE,
                       I_ORIGINAL, J_ORIGINAL, LR_ORIGINAL, NUM_KPTS_ORIGINAL,
                       NUM_KPTS_ORIGINAL_NONECK, I_ORIGINAL_NONECK,
                       J_ORIGINAL_NONECK, LR_ORIGINAL_NONECK)

sys.path.insert(0, '/Users/Robert/Documents/Caltech/CS81_Depth_Research/models/3d-pose-baseline/src')
from viz import show3Dpose
from scipy.misc import imread


################################################################################
# VISUALIZE HITS
################################################################################

def visualize_HIT(hit, mode='groundtruth'):
    image_filepath = os.path.join(HUMAN_IMAGES_DIR, hit['images_truth']['filename'])
    if not os.path.isfile(image_filepath):
        print "Could not find {}. Please download from the server.".format(image_filepath)
        return

    img = mpimg.imread(image_filepath)
    fig, ax = plt.subplots(1)
    imgplot = ax.imshow(img)

    xs = hit['annotations_truth']['kpts_2d'][::2]
    ys = hit['annotations_truth']['kpts_2d'][1::2]
    
    if mode == 'turkerorder':
        order = hit['trials'][0]['kpts_relative_depth']
        pt_anns = [order.index(i) for i in range(len(order))]
        plt.title("Turker Ordering")
    elif mode == 'groundtruth':
        order = hit['annotations_truth']['kpts_relative_depth']
        pt_anns = [order.index(i) for i in range(len(order))]
        plt.title("Ground Truth Ordering")
        # zs = hit['annotations_truth']['kpts_3d'][2::3]
    elif mode == 'coords':
        order = hit['annotations_truth']['kpts_relative_depth']
        order = [order.index(i) for i in range(len(order))]
        coords = center_data(hit['annotations_truth']['kpts_3d'])
        coords = [coords[i + 2] for i in range(0, len(coords), 3)]
        pt_anns = zip(order, coords)
    else:
        print "visualize_HIT: Don't know what that mode is. Returning."
        return
        
    for x, y, text in zip(xs, ys, pt_anns):
        t = ax.text(x, y, str(text), color="red", fontsize=12, size='smaller')
        t.set_bbox(dict(facecolor='green', alpha=0.5))
        circ = Circle((x, y), 3)
        ax.add_patch(circ)

def center_data(coords):
    # Subtract out the anchor keypoint
    LEFT_HIP_INDEX = 7
    z = coords[LEFT_HIP_INDEX * 3 + 2]
    return coords - np.tile([0, 0, z], 13)

def project_to_axis(pts, plane):
    pts = list(pts)
    if plane == 'x':
        del pts[0::3]
    elif plane == 'y':
        del pts[1::3]
    elif plane == 'z':
        del pts[2::3]
    return pts

def plot_baseline_viz(pts, dataset='baseline'):
    '''
    A small wrapper function around the visualizer from the baseline model.
    It visualizes just the sticks (skeleton) and a ground.

    Enables different datasets to be visualized. Must pass in skeleton
    representation with I and J, and the left-right distinguishing list.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if dataset == 'baseline':
        show3Dpose(np.array(pts), ax)
    elif dataset == 'original':
        show3Dpose(np.array(pts), ax, I=I_ORIGINAL, J=J_ORIGINAL, LR=LR_ORIGINAL)

def plot_3d_stickcoords(pts, dataset='original'):
    if dataset == 'baseline':
        I = I_BASELINE
        J = J_BASELINE
    elif len(pts) / 3 == NUM_KPTS_ORIGINAL and dataset == 'original':
        I = I_ORIGINAL
        J = J_ORIGINAL
    elif len(pts) / 3 == NUM_KPTS_ORIGINAL_NONECK and dataset == 'original':
        I = I_ORIGINAL_NONECK
        J = J_ORIGINAL_NONECK
    ann_inds = list(set(I).union(set(J)))

    xs, ys, zs = [], [], []
    for line in zip(I, J):
        x, y, z = [], [], []
        x.append(pts[line[0]*3])
        y.append(pts[line[0]*3 + 1])
        z.append(pts[line[0]*3 + 2])
        x.append(pts[line[1]*3])
        y.append(pts[line[1]*3 + 1])
        z.append(pts[line[1]*3 + 2])
        xs.append(x)
        ys.append(y)
        zs.append(z)

    trimmed = []
    [trimmed.extend(list(pts[ai*3:ai*3+3])) for ai in ann_inds]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = np.array(trimmed[0::3]), np.array(trimmed[1::3]), np.array(trimmed[2::3])
    ax.scatter(X, Y, Z)

    for x, y, z in zip(xs, ys, zs):
        ax.plot(x, y, z)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

def plot_2d_stickcoords(pts, plane='z', dataset='original'):
    '''
    plane - setting this is only to change the axis labels.
    '''
    if dataset == 'baseline':
        I = I_BASELINE
        J = J_BASELINE
    elif len(pts) / 2 == NUM_KPTS_ORIGINAL and dataset == 'original':
        I = I_ORIGINAL
        J = J_ORIGINAL
    elif len(pts) / 2 == NUM_KPTS_ORIGINAL_NONECK and dataset == 'original':
        I = I_ORIGINAL_NONECK
        J = J_ORIGINAL_NONECK
    ann_inds = list(set(I).union(set(J)))

    xs, ys = [], []
    for line in zip(I, J):
        x, y = [], []
        x.append(pts[line[0]*2])
        y.append(pts[line[0]*2 + 1])
        x.append(pts[line[1]*2])
        y.append(pts[line[1]*2 + 1])
        xs.append(x)
        ys.append(y)

    trimmed = []
    [trimmed.extend(list(pts[ai*2:ai*2+2])) for ai in ann_inds]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    X, Y = np.array(trimmed[0::2]), np.array(trimmed[1::2])
    ax.scatter(X, Y)

    for x, y in zip(xs, ys):
        ax.plot(x, y)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb in zip(Xb, Yb):
       ax.plot([xb], [yb], 'w')

    if plane == 'x':
        ax.set_xlabel('Y Label')
        ax.set_ylabel('Z Label')
    elif plane == 'y':
        ax.set_xlabel('X Label')
        ax.set_ylabel('Z Label')
    else:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')


def plot_image(kpts_2d, image_filename, label=None, **kwargs):
    '''
    Display the image with the keypoints marked

    label: image is annotated with different labels depending on value of label
        - None: No labels printed
        - "relative_depth": The relative depth ordering is printed from closest
          to the camera to furthest.
    '''
    if label == "relative_depth" and "kpts_relative_depth" not in kwargs:
        print ("kpts_relative_depth must be passed to plot_image.")
        return False
    elif label == "absolute_depth" and "kpts_3d" not in kwargs:
        print ("kpts_3d must be passed to plot_image.")
        return False

    try:
        read_pic = imread(HUMAN_IMAGES_DIR + '/{}'.format(image_filename))
    except IOError:
        print "Download {}".format(image_filename)
        return False
    plt.figure()
    plt.imshow(read_pic)
    xs = kpts_2d[0::2]
    ys = kpts_2d[1::2]
    plt.scatter(xs, ys)

    if label == "relative_depth":
        kpts_relative_depth = kwargs['kpts_relative_depth']
        for kpt_id, (x, y) in enumerate(zip(xs, ys)):
            depth_rank = kpts_relative_depth.index(kpt_id)
            t = plt.text(x, y, str(depth_rank), color="red", fontsize=12, size='smaller')
            t.set_bbox(dict(facecolor='green', alpha=0.5))
    
    elif label == "absolute_depth":
        kpts_3d = kwargs['kpts_3d']
        depths = map(int, kpts_3d[1::3])
        for kpt_id, (x, y) in enumerate(zip(xs, ys)):
            t = plt.text(x, y, str(depths[kpt_id]), color="red", fontsize=12, size='smaller')
            t.set_bbox(dict(facecolor='green', alpha=0.5))
    return True


# def plot_image(annotation, image, groundtruth):
#     # print groundtruth
#     ys = groundtruth[1::3]
#     print len(ys)
#     sortedys = sorted(ys, reverse=True)

#     H36M_NAMES = ['']*32
#     H36M_NAMES[0]  = 'Hip'
#     H36M_NAMES[1]  = 'RHip'
#     H36M_NAMES[2]  = 'RKnee'
#     H36M_NAMES[3]  = 'RFoot'
#     H36M_NAMES[6]  = 'LHip'
#     H36M_NAMES[7]  = 'LKnee'
#     H36M_NAMES[8]  = 'LFoot'
#     H36M_NAMES[12] = 'Spine'
#     H36M_NAMES[13] = 'Thorax'
#     H36M_NAMES[14] = 'Neck/Nose'
#     H36M_NAMES[15] = 'Head'
#     H36M_NAMES[17] = 'LShoulder'
#     H36M_NAMES[18] = 'LElbow'
#     H36M_NAMES[19] = 'LWrist'
#     H36M_NAMES[25] = 'RShoulder'
#     H36M_NAMES[26] = 'RElbow'
#     H36M_NAMES[27] = 'RWrist'
    
#     sorted_names = []
#     for i, name in enumerate(H36M_NAMES):
#         sorted_names.append(sortedys.index(ys[i]))
#         print name + " " + str(sortedys.index(ys[i]))
#     sorted_names = sorted(sorted_names)
#     print sorted_names

#     # print len(annotation['kpts_2d'])
#     # pt_anns = [order.index(i) for i in range(len(order))]

#     image_path = image['filename']
#     plt.figure()
#     plt.imshow(imread(HUMAN_IMAGES_DIR + '/%s'%image_path))
#     plt.scatter(annotation['kpts_2d'][0::2],annotation['kpts_2d'][1::2])

#     images = ["human36m_train_0000872034.jpg", "human36m_train_0000860493.jpg",
#               "human36m_train_0000963027.jpg", "human36m_train_0000074377.jpg",
#               "human36m_train_0000626565.jpg", "human36m_train_0000876351.jpg"]
