import os
import numpy as np

################################################################################
# CONSTANTS
################################################################################

RESULT_DIR = "/Users/Robert/Documents/Caltech/CS81_Depth_Research/results"
HUMAN_ANNOTATION_DIR = '/Users/Robert/Documents/Caltech/CS81_Depth_Research/' \
                       'datasets/human36m_annotations'
COCO_ANNOTATION_DIR = '/Users/Robert/Documents/Caltech/CS81_Depth_Research/'  \
                      'datasets/mscoco_annotations'
# HUMAN_ANNOTATION_DIR = '/home/ubuntu/datasets/human3.6/annotations'
# COCO_ANNOTATION_FILE = '/home/ubuntu/datasets/mscoco/annotations'
# HUMAN_IMAGES_DIR = '/Users/Robert/Documents/Caltech/CS81_Depth_Research/' \
#                    'datasets/human36m_images'
HUMAN_IMAGES_DIR = '/Users/Robert/Documents/Caltech/CS81_Depth_Research/' \
                   'datasets/human36m_images_17'

HUMAN_RAW_RESULT_FILE = "cocoa_test_completed_1000_DepthHITS_2018-04-24_09-14-49.pkl"
COCO_RAW_RESULT_FILE = "cocoa_test_completed_2400_DepthHITS_2017-06-13_13-22-13.pkl"
# HUMAN_ANNOTATION_FILE = 'human36m_train.json'
HUMAN_ANNOTATION_FILE = 'human36m_train_17.json'
COCO_ANNOTATION_FILE = 'person_keypoints_train2014.json'

HUMAN_ANNOTATION_PATH = os.path.join(HUMAN_ANNOTATION_DIR, HUMAN_ANNOTATION_FILE)
COCO_ANNOTATION_PATH = os.path.join(COCO_ANNOTATION_DIR, COCO_ANNOTATION_FILE)
HUMAN_RAW_RESULT_PATH = os.path.join(RESULT_DIR, HUMAN_RAW_RESULT_FILE)
COCO_RAW_RESULT_PATH = os.path.join(RESULT_DIR, COCO_RAW_RESULT_FILE)

KEYPTS_RELATIVE_DEPTH_FILE = "human36m_601_keypts_relative_depth.json"
KEYCMPS_RESULT_FILE = "human36m_601_keycmps.json"
KEYPTS_RELATIVE_DEPTH_PATH = os.path.join(RESULT_DIR, KEYPTS_RELATIVE_DEPTH_FILE)
KEYCMPS_RESULT_PATH = os.path.join(RESULT_DIR, KEYCMPS_RESULT_FILE)

HUMAN_OUTPUT_FILE = "human36m_processed_data.json"
HUMAN_OUTPUT_PATH = os.path.join(RESULT_DIR, HUMAN_OUTPUT_FILE)

CALTECH_OUTPUT_FILE = "human36m_processed_caltech_data.json"
CALTECH_OUTPUT_PATH = os.path.join(RESULT_DIR, CALTECH_OUTPUT_FILE)



# INTEREST_KEYPOINTS = \
#   ['head', 'neck',
#    'left_shoulder', 'right_shoulder',
#    'left_elbow', 'right_elbow',
#    'left_wrist', 'right_wrist',
#    'left_hip', 'right_hip',
#    'left_knee', 'right_knee',
#    'left_ankle', 'right_ankle']
# SKELETON = \
#   [['head', 'neck'],
#    ['neck', 'left_shoulder'], ['neck', 'right_shoulder'],
#    ['left_shoulder', 'left_elbow'], ['right_shoulder', 'right_elbow'],
#    ['left_elbow', 'left_wrist'], ['right_elbow', 'right_wrist'],
#    ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
#    ['left_hip', 'left_knee'], ['right_hip', 'right_knee'],
#    ['left_knee', 'left_ankle'], ['right_knee', 'right_ankle']]
NUM_KPTS_EXPANDED = 17
I_ORIGINAL  = np.array([1,2,2,3,4,5,6,3,4 ,9 ,10,11,12])-1 # start points
J_ORIGINAL  = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14])-1 # end points
LR_ORIGINAL = np.array([1,1,0,1,0,1,0,1,0,1 ,0 ,1 ,0 ], dtype=bool)

NUM_KPTS_ORIGINAL = 14
I_ORIGINAL  = np.array([1,2,2,3,4,5,6,3,4 ,9 ,10,11,12])-1 # start points
J_ORIGINAL  = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14])-1 # end points
LR_ORIGINAL = np.array([1,1,0,1,0,1,0,1,0,1 ,0 ,1 ,0 ], dtype=bool)

NUM_KPTS_ORIGINAL_NONECK = 13
I_ORIGINAL_NONECK  = np.array([1,1,2,3,4,5,2,3 ,8 ,9,10,11])-1 # start points
J_ORIGINAL_NONECK  = np.array([2,3,4,5,6,7,8,9,10,11,12,13])-1 # end points
LR_ORIGINAL_NONECK = np.array([1,0,1,0,1,0,1,0,1 ,0 ,1 ,0 ], dtype=bool)


#####################################
# 3d pose baseline model constants

POSE_BASELINE_MODEL_PATH = "/Users/Robert/Documents/Caltech/" \
                           "CS81_Depth_Research/models/3d-pose-baseline"
BASELINE_DATA_PATH = "/Users/Robert/Documents/Caltech/CS81_Depth_Research/models/3d-pose-baseline/data/h36m"

I_BASELINE  = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
J_BASELINE  = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
LR_BASELINE = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)


##############################
# MATLAB FILES

ROTATION_MATRICES_PATH = "/Users/Robert/Documents/Caltech/CS81_Depth_Research/" \
                         "datasets/human36m_original/Release-v1.1/H36M/" \
                         "rotationmatrices.mat"

CAMERA_NAMES_PATH = "/Users/Robert/Documents/Caltech/CS81_Depth_Research/" \
                    "datasets/human36m_original/Release-v1.1/H36M/" \
                    "cameranames.mat"

