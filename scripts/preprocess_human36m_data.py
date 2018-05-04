## imports
from os import listdir
import json, sys
import numpy as np
import random
random.seed(17)
#import skvideo.io
import imageio
from  scipy.misc import imresize

from spacepy import pycdf
# import cdflib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

## set paths
DATASET_DIR      = '../datasets/human36m_original'
ANNOTATIONS_DIR  = '../datasets/human36m_annotations'
IMAGES_DIR       = '../datasets/human36m_images'

## dataset constants
FEATURE_TYPES = ['D2_Positions','D3_Positions_mono', 'D3_Positions_mono_universal']
# FEATURE_TYPES = ['D2_Positions']
IMAGES_WIDTH  = 256
IMAGES_HEIGHT = 256
SKIP_FRAMES   = 20
SKIP_ACTIONS  = ['_ALL']#,'WalkTogether','Posing','Photo','SittingDown','Directions','Purchases','Sitting','Walking','Waiting','Phoning','Smoking','WalkDog','Discussion','Eating']
SKIP_ACTIONS  = ['_ALL', 'WalkTogether','Posing','Photo','SittingDown','Directions','Purchases','Sitting','Walking','Waiting','Phoning','Smoking','WalkDog','Discussion','Eating', 'WalkingDog', 'Greeting']
#CAMERA_IDS    = [54138969, 55011271, 58860488, 60457274]
SUBJECT_IDS   = [1,5,6,7,8,9,11]
# SUBJECT_IDS   = [1]
HUMAN_36M_KEYPOINTS = \
  ['mid_hip',
   'right_hip', 'right_knee', 'right_ankle', 'right_foot_base', 'right_foot_tip',
   'left_hip', 'left_knee', 'left_ankle', 'left_foot_base', 'left_foot_tip',
   'mid_hip_2', 'mid_spine', 'neck', 'chin', 'head', 'neck_2',
   'left_shoulder', 'left_elbow', 'left_wrist', 'left_wrist_2','left_palm','left_thumb','left_thumb_2',
   'neck_3',
   'right_shoulder', 'right_elbow', 'right_wrist', 'right_wrist_2','right_palm','right_thumb','right_thumb_3']
# INTEREST_KEYPOINTS = \
#   ['head', 'neck',
#    'left_shoulder', 'right_shoulder',
#    'left_elbow', 'right_elbow',
#    'left_wrist', 'right_wrist',
#    'left_hip', 'right_hip',
#    'left_knee', 'right_knee',
#    'left_ankle', 'right_ankle']
INTEREST_KEYPOINTS = \
  ['head', 'neck',
   'left_shoulder', 'right_shoulder',
   'left_elbow', 'right_elbow',
   'left_wrist', 'right_wrist',
   'left_hip', 'right_hip',
   'left_knee', 'right_knee',
   'left_ankle', 'right_ankle',
   'mid_hip', 'mid_spine', 'chin']
SKELETON = \
  [['head', 'neck'],
   ['neck', 'left_shoulder'], ['neck', 'right_shoulder'],
   ['left_shoulder', 'left_elbow'], ['right_shoulder', 'right_elbow'],
   ['left_elbow', 'left_wrist'], ['right_elbow', 'right_wrist'],
   ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
   ['left_hip', 'left_knee'], ['right_hip', 'right_knee'],
   ['left_knee', 'left_ankle'], ['right_knee', 'right_ankle']]

## create json annotations for the human3.6m dataset for the specified settings
human36m = {}
human36m['annotations'] = []
human36m['images']      = []
human36m['actions']     = []
human36m['pose']        = []

annotation = {}
annotation['id']   = -1
annotation['s_id'] = -1
annotation['a_id'] = -1
annotation['i_id'] = -1
annotation['kpts_2d'] = []
annotation['kpts_3d'] = []

image = {}
image['id']       = -1
image['c_id']     = -1
image['filename'] = ''
image['width']    = IMAGES_WIDTH
image['height']   = IMAGES_HEIGHT
image['video']    = ''
image['frame']    = -1

actions = {}
actions['id']       = -1
actions['name']     = ''
actions['version']  = ''

pose = {}
pose['keypoints']      = INTEREST_KEYPOINTS
pose['skeleton']       = [map(INTEREST_KEYPOINTS.index,t) for t in SKELETON]
pose['original_index'] = [HUMAN_36M_KEYPOINTS.index(k) for k in INTEREST_KEYPOINTS]
human36m['pose'] = [pose]

PADDED   = []
USED_IDS = set()

for subject_id in SUBJECT_IDS:
    VIDEOS_DIR   = '%s/S%d/MyVideos'%(DATASET_DIR, subject_id)
    FEATURES_DIR = '%s/S%d/MyPoseFeatures'%(DATASET_DIR, subject_id)

    VIDEOS = listdir(VIDEOS_DIR)
    for video_filename in VIDEOS:
        video_info  = video_filename.split('.')
        action_info = video_info[0].split(' ')

        camera_id   = int(video_info[1])
        action_name = action_info[0]
        action_version = int(action_info[1]) if len(action_info) > 1 else 0

        if action_name in SKIP_ACTIONS: continue 
        if subject_id == 9 and camera_id == 58860488: continue
        print("S%d"%subject_id, action_name, action_version, camera_id)

        # insert the action in the actions list if it has never been encountered
        new_action = [a for a in human36m['actions'] if \
                        a['name'] == action_name and \
                        a['version'] == action_version]
        if len(new_action)==0:
            action_id = len(human36m['actions'])
            action = {}
            action['id']       = action_id
            action['name']     = action_name
            action['version']  = action_version
            human36m['actions'].append(action)
        else:
            action_id = new_action[0]['id']

        # extract the image frames
        frames = imageio.get_reader(VIDEOS_DIR + '/' + video_filename,  'ffmpeg')
        print("Frames %d"%(len(frames)))

        features = []
        shapes   = []
        # extract the features
        for feature_type in FEATURE_TYPES:
            print(FEATURES_DIR + '/' + feature_type + '/' + '.'.join(video_info[:-1]) + '.cdf')
            # cdf = cdflib.CDF(FEATURES_DIR + '/' + feature_type + '/' + '.'.join(video_info[:-1]) + '.cdf')
            cdf = pycdf.CDF(FEATURES_DIR + '/' + feature_type + '/' + '.'.join(video_info[:-1]) + '.cdf')
            # posedata = cdf.varget('Pose')[0,:,:][...]
            # features.append(np.reshape(posedata.T, posedata.shape))
            # features.append(cdf.varget('Pose')[0,:,:][...])
            features.append(cdf['Pose'][0,:,:][...])

            # shape   = cdf.varget('Pose')[0,:,:][...].shape
            shape   = cdf['Pose'][0,:,:][...].shape
            shapes.append(shape[0])
            print(feature_type, shape)

        assert(shapes.count(shapes[0]) == len(shapes))
        print("=================================")

#         frame_num = 0
#         # access features and image one frame at the time
#         # NOTE: number of frames and number of features might be different!
#         # assumption is that they are alligned at beginning and the final
#         # frames get discarded.
#         while frame_num < shapes[0]:
#             frame   = frames.get_data(frame_num)

#             # 2d pose associated with that frame
#             pose_2d = features[FEATURE_TYPES.index('D2_Positions')][frame_num,:]
#             pose_2d_x = pose_2d[0::2]
#             pose_2d_y = pose_2d[1::2]

#             w  = max(pose_2d_x) - min(pose_2d_x)
#             h  = max(pose_2d_y) - min(pose_2d_y)
#             cx = int(min(pose_2d_x) + w/2.)
#             cy = int(min(pose_2d_y) + h/2.)

#             bbox = [cx - (w*1.2)/2., cy - (h*1.2)/2., w*1.2, h*1.2] # 20% enlarged
#             slack = int(bbox[2]/2.) if w > h else int(bbox[3]/2.)
#             x_start = cx - slack
#             x_end   = cx + slack
#             y_start = cy - slack
#             y_end   = cy + slack

#             print(pose_2d_x)
#             print(pose_2d_y)
#             print(cx, cy, w, h)
#             print(bbox)
#             print(slack)
#             print(x_start, x_end)
#             print(y_start, y_end)

#             pad_left   = abs(x_start) if x_start < 0 else 0
#             #pad_right  = x_end - 2 * slack if x_end > 2 * slack else 0
#             pad_top    = abs(y_start) if y_start < 0 else 0
#             # pad_bottom = y_end - 2 * slack if y_end > 2 * slack else 0
#             padded_frame = np.pad(frame,((0,0),(pad_left,0),(0,0)),'edge')
#             # try:
#             crop = imresize(padded_frame[y_start+pad_top:y_end+pad_top, x_start+pad_left:x_end+pad_left, :],(IMAGES_WIDTH, IMAGES_WIDTH))
#             # except:
#             #     print frame.shape
#             #     print cx, x_start, x_end, 2*slack
#             #     print cy, y_start, y_end, 2*slack
#             #     print pad_top, pad_bottom, pad_left, pad_top
#             #     assert(False)

#             resize_ratio = [IMAGES_WIDTH / (2. * slack), IMAGES_HEIGHT / (2. * slack)]
#             keypoints_2d = []
#             for i in human36m['pose'][0]['original_index']:
#                 pose_2d_x_i = (pose_2d_x[i] - x_start) * resize_ratio[0]
#                 pose_2d_y_i = (pose_2d_y[i] - y_start) * resize_ratio[1]
#                 keypoints_2d.extend([pose_2d_x_i,pose_2d_y_i])

#             # fig = plt.figure()
#             # ax = plt.subplot(121)
#             # ax.imshow(frame)
#             # ax.scatter(pose_2d_x,pose_2d_y,c='g',s=50)
#             # ax.scatter(cx,cy,c='r',s=50)
#             # rect  = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor='b',facecolor='none')
#             # ax.add_patch(rect)
#             # rect1  = patches.Rectangle((x_start,y_start),2*slack,2*slack,linewidth=2,edgecolor='y',facecolor='none')
#             # ax.add_patch(rect1)
#             # ax = plt.subplot(122)
#             # ax.imshow(crop)
#             # ax.scatter(keypoints_2d[0::2],keypoints_2d[1::2],c='g',s=50)
#             # plt.show()

#             # 3d pose associated with that frame
#             pose_3d = features[FEATURE_TYPES.index('D3_Positions_mono')][frame_num,:]
#             pose_3d_x = pose_3d[0::3]
#             pose_3d_y = pose_3d[1::3]
#             pose_3d_z = pose_3d[2::3]
#             keypoints_3d = []
#             for i in human36m['pose'][0]['original_index']:
#                 keypoints_3d.extend([pose_3d_x[i],pose_3d_y[i],pose_3d_z[i]])
#             # fig = plt.figure()
#             # ax = fig.add_subplot(111, projection='3d')
#             # ax.scatter(keypoints_3d[0::3],keypoints_3d[1::3],keypoints_3d[2::3])
#             # plt.show()

#             rand_id       = random.randint(0,999999)
#             while rand_id in USED_IDS:
#                 rand_id       = random.randint(0,999999)
#             USED_IDS.add(rand_id)

#             annotation_id = rand_id
#             image_id      = rand_id

#             annotation = {}
#             annotation['id']   = annotation_id
#             annotation['s_id'] = subject_id
#             annotation['a_id'] = action_id
#             annotation['i_id'] = image_id
#             annotation['kpts_2d'] = map(int,keypoints_2d)
#             annotation['kpts_3d'] = map(int,keypoints_3d)
#             human36m['annotations'].append(annotation)

#             image = {}
#             image['id']       = image_id
#             image['c_id']     = camera_id
#             image['s_id']     = subject_id
#             image['filename'] = 'human36m_train_%010d.jpg'%(image_id)
#             image['width']    = IMAGES_WIDTH
#             image['height']   = IMAGES_HEIGHT
#             image['video']    = video_filename
#             image['frame']    = frame_num
#             human36m['images'].append(image)

# #             # save the image
# #             image_filename = '%s/human36m_train_%010d.jpg'%(IMAGES_DIR,image_id)
# #             imageio.imwrite(image_filename, crop)

#             if pad_left + pad_top != 0:
#                 PADDED.append(image_id)
#             frame_num += SKIP_FRAMES

#         # close the frame reader before processing next action
#         frames.close()

# # with open('%s/human36m_train.json'%(ANNOTATIONS_DIR),'wb') as fp:
# #     json.dump(human36m,fp)

# # print human36m.keys()
# # print len(human36m['annotations'])
# # print len(human36m['images'])

# # print human36m['pose']
# # print human36m['actions']

# # with open('./padded.json','wb') as fp:
# #     json.dump(PADDED, fp)

# # print len(PADDED), PADDED
