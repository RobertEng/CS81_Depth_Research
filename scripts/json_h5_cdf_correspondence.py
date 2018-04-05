import h5py
from spacepy import pycdf
import json
import numpy as np

"""
################################################################################
DATA REPRESENTED IN OUR FORMAT
################################################################################
"""
human_36m_file_path = '/home/mronchi/Storage/Datasets/human3.6/annotations/human36m_train.json'
with open(human_36m_file_path,'rb') as fp:
    data = json.load(fp)

anns    = data['annotations']
imgs    = data['images']
actions = data['actions']

RANDOM_ANNOTATION = np.random.randint(0, len(anns))
ann = anns[RANDOM_ANNOTATION]
img = [i for i in imgs if i['id']==ann['i_id']][0]

subj   = img['s_id']
frame  = img['frame']
name   = img['video'].split(".")[0]
camera = img['video'].split(".")[1]

our_keypoints = ann['kpts_3d']

"""
################################################################################
DATA REPRESENTED IN HUMAN3.6m FORMAT
################################################################################
"""
cdf_file_path = '/home/mronchi/Storage/Datasets/human3.6/Release-v1.1/data/S%d/MyPoseFeatures/D3_Positions_mono/%s.%s.cdf'%(subj,name,camera)
# pose of the subject for all frames in video in camera 1 coordinates
cdf_file = pycdf.CDF(cdf_file_path)
# numpy array of 96 x num_frames; 96 is 3 x num_keypoints; human_36 has 32 keypoints
c_poses  = cdf_file['Pose'][:]

camera_keypoints = c_poses[:,frame]

"""
################################################################################
DATA REPRESENTED IN 3D BASELINE PAPER FORMAT
################################################################################
"""
h5_file_path_w   = '/home/mronchi/Research/Projects/keypoints-3d-detection/3d-pose-baseline/data/h36m/S%d/MyPoses/3D_positions/%s.h5'%(subj,name)
hf_cameras_path  = '/home/mronchi/Research/Projects/keypoints-3d-detection/3d-pose-baseline/data/h36m/cameras.h5'

# pose of the subject for all frames in video in world coordinates
h5_file  = h5py.File(h5_file_path_w,'r')
# numpy array of 96 x num_frames; 96 is 3 x num_keypoints; human_36 has 32 keypoints
w_poses   = h5_file['3D_positions'][:]

world_keypoints = w_poses[:,frame]

print "OUR   : [%s]"%(our_keypoints)
print "CAMERA: [%s]"%(camera_keypoints)
print "WORLD : [%s]"%(world_keypoints)
