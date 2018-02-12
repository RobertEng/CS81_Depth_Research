import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

import postprocess_original_utils
import postprocess_baseline_utils
from viz_utils import (visualize_HIT, plot_image, plot_baseline_viz,
    plot_3d_stickcoords, project_to_axis, plot_2d_stickcoords)
from constants import (HUMAN_ANNOTATION_PATH, HUMAN_IMAGES_DIR,
    POSE_BASELINE_MODEL_PATH, BASELINE_DATA_PATH)

sys.path.insert(0, '/Users/Robert/Documents/Caltech/CS81_Depth_Research/models/3d-pose-baseline/src')
import cameras
import data_utils


version = 0
# i_human = 0
# i_baseline = 0
i = 0
# i_baseline = i_human * 5 ## DOESN"T WORK. EXAMINE THIS LATER.

SUBJ_IDS = [1]
camera_frame = False
experimental = False
actions = ['Directions']
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
# # print(data[536]['annotations_truth'])
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

from constants import CALTECH_OUTPUT_PATH
data = json.load(open(CALTECH_OUTPUT_PATH, 'r'))
data = [d for d in data if d['_worker_id'] == 'nonAMT_368102']
'''
{u'Amanda Lin: nonAMT_808135',
 u'Caltech: ',
 u'Jennifer - Hi! I wou: nonAMT_607266',
 u'Matteo: nonAMT_368102',
 u'Milan: nonAMT_6599',
 u'Oisin: nonAMT_687008',
 u'Ron: nonAMT_764039',
 u'This is Jalani. I wo: nonAMT_158235',
 u'caltech: ',
 u'lucy: nonAMT_700986'}

img_ids = [56833, 965922, 849671, 649263, 12750, 896082, 571176, 965922]
 '''
img_id = 56833 # easy
img_id = 965922 # easy
img_id = 849671 # easy-medium
img_id = 649263 # medium
img_id = 12750 # hard
img_id = 896082 # hard
img_id = 571176 # medium

for _i, d in enumerate(data):
    if d['annotations_truth']['i_id'] == img_id:
        idx = _i
        break
print idx
i = idx

plot_image(data[i]['annotations_truth']['kpts_2d'],
           data[i]['images_truth']['filename'], label="relative_depth",
           kpts_relative_depth=data[i]['annotations_truth']['kpts_relative_depth'])
plt.title("Ground Truth Ordering")
plot_image(data[i]['annotations_truth']['kpts_2d'],
           data[i]['images_truth']['filename'], label="relative_depth",
           kpts_relative_depth=data[i]['kpts_relative_depth'])
plt.title("Turker Guess Ordering")
plot_image(data[i]['annotations_truth']['kpts_2d'],
           data[i]['images_truth']['filename'], label="absolute_depth",
           kpts_3d=data[i]['annotations_truth']['kpts_3d'])
plt.title("Ground Truth Depth")

plot_3d_stickcoords(data[i]['annotations_truth']['kpts_3d'], dataset='original')

plane = 'z'
pts_projected = project_to_axis(data[i]['annotations_truth']['kpts_3d'], plane)
plot_2d_stickcoords(pts_projected, plane=plane)

plt.show()



# data_by_hit = postprocess_original_utils.load_data()
# # data_by_img = group_data_by_image(data_by_hit)

# images = ["human36m_train_0000044510.jpg", "human36m_train_0000408524.jpg",
#           "human36m_train_0000668287.jpg", "human36m_train_0000167841.jpg",
#           "human36m_train_0000465672.jpg"]

# hits = postprocess_original_utils.lookup_hits_from_file_names(data_by_hit, images)

# for h in hits:
#     visualize_HIT(h, mode='coords')
#     # visualize_HIT(h, mode='groundtruth')
# plt.show()

