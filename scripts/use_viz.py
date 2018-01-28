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

with open(HUMAN_ANNOTATION_PATH, 'rb') as fp:
    _human_dataset = json.load(fp)

# img_id = 44510   # hallwaybg, chair sitter
# img_id = 465672  # hallwaybg, normal stander
# img_id = 668287  # projectorbg, sass lady
# img_id = 167841  # projectorbg, sadwalker
# img_id = 521983  # radiatorbg, OG im flying pose
# img_id = 950012  # radiatorbg, phone sitter
# img_id = 223040  # doorbg, back dude
# img_id = 708446  # doorbg, back sitter dude
# img_id = 408524  # hallwaybg, OG im flying pose
img_id = 790655  # annotation 0 for s1
for i, d in enumerate(_human_dataset['annotations']):
    if d['i_id'] == img_id:
        idx = i
        break
print idx

version = 0
i_human = 0
i_baseline = 0
# i_baseline = i_human * 5 ## DOESN"T WORK. EXAMINE THIS LATER.

SUBJ_IDS = [1]
camera_frame = False
experimental = False
actions = ['Directions']
# actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo',
#     'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Waiting',
#     'WalkDog', 'Walking', 'WalkTogether']


## BASELINE VIZ
# baseline_data = data_utils.load_data(BASELINE_DATA_PATH, SUBJ_IDS, actions)
# if camera_frame:
#     rcams = cameras.load_cameras(os.path.join(POSE_BASELINE_MODEL_PATH, "data/h36m/cameras.h5"), SUBJ_IDS)
#     baseline_data = data_utils.transform_world_to_camera( baseline_data, rcams, experimental=experimental )
# action_data = postprocess_baseline_utils.get_version(baseline_data, actions, version)

# # plot_baseline_viz(action_data[i_baseline], dataset='baseline')
# plot_3d_stickcoords(action_data[i_baseline], dataset='baseline')

# plane = 'y'
# pts_projected = project_to_axis(action_data[i_baseline], plane)
# plot_2d_stickcoords(pts_projected, plane=plane, dataset='baseline')



## ORIGINAL VIZ
action_data_annotations, action_data_images = postprocess_original_utils.get_action_data_from_human(_human_dataset, SUBJ_IDS, actions[0], version)
print action_data_annotations[i_human]['kpts_3d']
print action_data_images[i_human]['filename']

# plot_image(action_data_annotations[i_human], action_data_images[i_human], action_data[i_baseline])
# plot_baseline_viz(_human_dataset['annotations'][idx]['kpts_3d'], dataset='original')
plot_3d_stickcoords(action_data_annotations[i_human]['kpts_3d'], dataset='original')

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

