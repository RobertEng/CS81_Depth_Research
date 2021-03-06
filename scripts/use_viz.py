import os
import sys
import json
import random
random.seed(420)
import matplotlib.pyplot as plt
import numpy as np

import postprocess_original_utils
import postprocess_baseline_utils
from viz_utils import (visualize_HIT, plot_image, plot_baseline_viz,
    plot_3d_stickcoords, project_to_axis, plot_2d_stickcoords)
from constants import (HUMAN_ANNOTATION_PATH, HUMAN_IMAGES_DIR,
    POSE_BASELINE_MODEL_PATH, BASELINE_DATA_PATH)
from lean_correction import correct_lean

sys.path.insert(0, '/Users/Robert/Documents/Caltech/CS81_Depth_Research/models/3d-pose-baseline/src')
import cameras
import data_utils


# version = 0
# i_human = 0
# i_baseline = 0
# i = 0
# i_baseline = i_human * 5 ## DOESN"T WORK. EXAMINE THIS LATER.

# SUBJ_IDS = [1]
# SUBJ_IDS = [279782]
# camera_frame = False
# experimental = False
# actions = ['Directions']
# actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo',
#     'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Waiting',
#     'WalkDog', 'Walking', 'WalkTogether']


################################################################################
## BASELINE VIZ
# baseline_data = data_utils.load_data(BASELINE_DATA_PATH, SUBJ_IDS, actions)
# if camera_frame:
#     rcams = cameras.load_cameras(os.path.join(POSE_BASELINE_MODEL_PATH, "data/h36m/cameras.h5"), SUBJ_IDS)
#     baseline_data = data_utils.transform_world_to_camera( baseline_data, rcams, experimental=experimental )
# action_data = postprocess_baseline_utils.get_version(baseline_data, actions, version)

# # plot_baseline_viz(action_data[i], dataset='baseline')
# plot_3d_stickcoords(action_data[i], dataset='baseline')

# plane = 'y'
# pts_projected = project_to_axis(action_data[i], plane)
# plot_2d_stickcoords(pts_projected, plane=plane, dataset='baseline')

################################################################################
## ORIGINAL VIZ
# idx = None
# img_id = 279782
# with open( HUMAN_ANNOTATION_PATH ) as f:
#   _human_dataset = json.load(f)
#   correct_lean(_human_dataset)
#   for _i, d in enumerate(_human_dataset['images']):
#     if d['id'] == img_id:
#       idx = _i
#       break
# print idx
# i = idx

# data = _human_dataset['annotations']
# plot_image(_human_dataset['annotations'][i]['kpts_2d'],
#            _human_dataset['images'][i]['filename'])
# # plot_3d_stickcoords(_human_dataset['annotations'][i]['kpts_3d'])
# plt.show()

def plot_image_from_subj_and_camera(_human_dataset, subj_id, camera_id, video=None, count=1):
  hdas, hdis = [], []
  for da, di in zip(_human_dataset['annotations'], _human_dataset['images']):
    if di['c_id'] == camera_id and da['s_id'] == subj_id:
      if video is None or video == di['video']:
        hdas.append(da)
        hdis.append(di)
  
  if len(hdas) == 0:
    print "No image for s_id {}, c_id {}".format(subj_id, camera_id)

  idxes = random.sample(range(len(hdas)), min(count, len(hdas)))
  for idx in idxes:
    if plot_image(hdas[idx]['kpts_2d'], hdis[idx]['filename']):
      print "s_id {}, c_id {}".format(subj_id, camera_id)
      plt.show()

with open( HUMAN_ANNOTATION_PATH ) as f:
  _human_dataset = json.load(f)
  correct_lean(_human_dataset)

def plot_bad_images():
  subj_id = 9
  for da, di in zip(_human_dataset['annotations'], _human_dataset['images']):
    if di['video'] == "SittingDown 1.54138969.mp4" and da['s_id'] == subj_id:
      if plot_image(da['kpts_2d'], di['filename']):
        print "s_id {}, c_id {}, video {}".format(subj_id, di['c_id'], di['video'])
        plt.show()
        break

def plot_all_cameras_and_subjs():
  camera_ids = [54138969, 55011271, 58860488, 60457274]
  subject_ids = [1,5,6,7,8,9,11]
  for s_id in subject_ids:
    for c_id in camera_ids:
      plot_image_from_subj_and_camera(_human_dataset, s_id, c_id)

def save_all_annotated():
  videos = {}
  for da, di in zip(_human_dataset['annotations'], _human_dataset['images']):
    subj_id = da['s_id']
    
    if (di['video'], subj_id) not in videos:
      # Save random frame on first encounter
      if plot_image(da['kpts_2d'], di['filename']):
        fname = '.'.join(di['video'].split('.')[:-1]) + "_" + str(subj_id) + "_" + str(di['frame']) + "_random_" + di['filename']
        plt.savefig('/Users/Robert/Documents/Caltech/CS81_Depth_Research/datasets/human36m_images_17_annotated/' + fname)
        plt.close()
      videos[(di['video'], subj_id)] = di['frame']
    
    if di['frame'] < videos[(di['video'], subj_id)]:
      videos[(di['video'], subj_id)] = di['frame']

  for da, di in zip(_human_dataset['annotations'], _human_dataset['images']):
    subj_id = da['s_id']
    if videos[(di['video'], subj_id)] == di['frame']:
      if plot_image(da['kpts_2d'], di['filename']):
        fname = '.'.join(di['video'].split('.')[:-1]) + "_" + str(subj_id) + "_" + str(di['frame']) + "_last_" + di['filename']
        plt.savefig('/Users/Robert/Documents/Caltech/CS81_Depth_Research/datasets/human36m_images_17_annotated/' + fname)
        plt.close()



for da, di in zip(_human_dataset['annotations'], _human_dataset['images']):
  if di['filename'] == 'human36m_train_0000055050.jpg':
    plot_image(da['kpts_2d'], 'human36m_train_0000055050.jpg')
    plt.show()

# plot_image_from_subj_and_camera(_human_dataset, 9, 54138969, video="SittingDown 1.54138969.mp4")
# plot_image_from_subj_and_camera(_human_dataset, 9, 55011271, video="SittingDown 1.55011271.mp4")
# plot_image_from_subj_and_camera(_human_dataset, 9, 58860488, video="SittingDown 1.58860488.mp4")
# plot_image_from_subj_and_camera(_human_dataset, 9, 60457274, video="SittingDown 1.60457274.mp4")

# plot_image_from_subj_and_camera(_human_dataset, 9, 54138969, count=5)
# plot_image_from_subj_and_camera(_human_dataset, 9, 58860488, count=5)



################################################################################
## ORIGINAL VIZ WITH TURKERS

# idx = None

# # TURKER DATA
# data = postprocess_original_utils.load_data()

# # NOTE TO SELF. Not all these images have been annotated by turkers, and thus
# # won't be viewable. Just don't freak out when it can't find the image.
# # img_id = 44510   # hallwaybg, chair sitter
# # img_id = 465672  # hallwaybg, normal stander
# # img_id = 668287  # projectorbg, sass lady
# # img_id = 167841  # projectorbg, sadwalker
# # img_id = 521983  # radiatorbg, OG im flying pose
# # img_id = 950012  # radiatorbg, phone sitter
# # img_id = 223040  # doorbg, back dude
# # img_id = 708446  # doorbg, back sitter dude
# # img_id = 408524  # hallwaybg, OG im flying pose
# # img_id = 790655  # annotation 0 for s1
# # img_id = 37859 # suspicious guy who annotated everything wrong but actually the pic is hard
# # img_id = 202669
# img_id = 279782
# # print(data[536])
# for _i, d in enumerate(data):
#     if d['annotations_truth']['i_id'] == img_id:
#         idx = _i
#         break
# print idx
# i = idx

# plot_image(data[i]['annotations_truth']['kpts_2d'],
#            data[i]['images_truth']['filename'], label="relative_depth",
#            kpts_relative_depth=data[i]['annotations_truth']['kpts_relative_depth'])
# plt.title("Ground Truth Ordering")
# plot_image(data[i]['annotations_truth']['kpts_2d'],
#            data[i]['images_truth']['filename'], label="relative_depth",
#            kpts_relative_depth=data[i]['trials'][0]['kpts_relative_depth'])
# plt.title("Turker Guess Ordering")
# plot_image(data[i]['annotations_truth']['kpts_2d'],
#            data[i]['images_truth']['filename'], label="absolute_depth",
#            kpts_3d=data[i]['annotations_truth']['kpts_3d'])
# plt.title("Ground Truth Depth")

# plot_3d_stickcoords(data[i]['annotations_truth']['kpts_3d'], dataset='original')

# plane = 'z'
# pts_projected = project_to_axis(data[i]['annotations_truth']['kpts_3d'], plane)
# plot_2d_stickcoords(pts_projected, plane=plane)

# plt.show()

################################################################################
# CALTECH VIZ

# from constants import CALTECH_OUTPUT_PATH
# data = json.load(open(CALTECH_OUTPUT_PATH, 'r'))
# data = [d for d in data if d['_worker_id'] == 'nonAMT_687008']
# '''
# {u'Amanda Lin: nonAMT_808135',
#  u'Caltech: ',
#  u'Jennifer - Hi! I wou: nonAMT_607266',
#  u'Matteo: nonAMT_368102',
#  u'Milan: nonAMT_6599',
#  u'Oisin: nonAMT_687008',
#  u'Ron: nonAMT_764039',
#  u'This is Jalani. I wo: nonAMT_158235',
#  u'caltech: ',
#  u'lucy: nonAMT_700986'}

# img_ids = [56833, 965922, 849671, 649263, 12750, 896082, 571176, 965922]
#  '''
# img_id = 56833 # easy
# # img_id = 965922 # easy
# # img_id = 849671 # easy-medium
# # img_id = 649263 # medium
# # img_id = 12750 # hard
# # img_id = 896082 # hard
# # img_id = 571176 # medium

# for _i, d in enumerate(data):
#     if d['annotations_truth']['i_id'] == img_id:
#         idx = _i
#         break
# print idx
# i = idx

# plot_image(data[i]['annotations_truth']['kpts_2d'],
#            data[i]['images_truth']['filename'], label="relative_depth",
#            kpts_relative_depth=data[i]['annotations_truth']['kpts_relative_depth'])
# plt.title("Ground Truth Ordering")
# plot_image(data[i]['annotations_truth']['kpts_2d'],
#            data[i]['images_truth']['filename'], label="relative_depth",
#            kpts_relative_depth=data[i]['trials'][0]['kpts_relative_depth'])
# plt.title("Turker Guess Ordering")
# plot_image(data[i]['annotations_truth']['kpts_2d'],
#            data[i]['images_truth']['filename'], label="absolute_depth",
#            kpts_3d=data[i]['annotations_truth']['kpts_3d'])
# plt.title("Ground Truth Depth")

# plot_3d_stickcoords(data[i]['annotations_truth']['kpts_3d'], dataset='original')

# plane = 'z'
# pts_projected = project_to_axis(data[i]['annotations_truth']['kpts_3d'], plane)
# plot_2d_stickcoords(pts_projected, plane=plane)

# plt.show()
