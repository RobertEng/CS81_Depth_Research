# Lean Correction
import math
import numpy as np
import scipy.io as sio

from constants import (ROTATION_MATRICES_PATH, CAMERA_NAMES_PATH)

def load_rotation_matrices():
    cam_info = sio.loadmat(ROTATION_MATRICES_PATH)['cam_info']
    # cameras = [1, 2, 3, 4]
    # SUBJECT_IDS   = [1,5,6,7,8,9,11]

    cam_info = np.rollaxis(cam_info, 3, 0)
    cam_info = np.rollaxis(cam_info, 3, 0)
    return cam_info

def load_camera_names():
    cam_names = sio.loadmat(CAMERA_NAMES_PATH)['cam_names']
    cameras = [1, 2, 3, 4]
    return cam_names

def correct_lean(_human_dataset):
    rotation_matrices = load_rotation_matrices()
    camera_names = load_camera_names()
    camera_names = [int(n) for n in camera_names]
    SUBJECT_IDS   = [1,5,6,7,8,9,11] # MUST MATCH MATLAB MATRIX

    # Associate the rotation matrix with each image
    # TODO: Refactor this loop out of this function
    for i in range(len(_human_dataset['images'])):
        cam_ind = camera_names.index(_human_dataset['images'][i]['c_id'])
        subj_ind = SUBJECT_IDS.index(_human_dataset['images'][i]['s_id'])
        _human_dataset['images'][i]["R"] = rotation_matrices[cam_ind][subj_ind].T.tolist()

    # Correct the lean
    for i in range(len(_human_dataset['images'])):
        kpts_3d = _human_dataset['annotations'][i]['kpts_3d']
        kpts_3d = np.reshape(kpts_3d, (len(kpts_3d) / 3, 3))

        # Unapply the rotation matrix
        kpts_3d = np.dot(kpts_3d, np.array(_human_dataset['images'][i]["R"]).T)
        
        # Convert the rotatioin matrix to euler angles and cancel all rotations
        # except those about the z axis (rotates person so they're facing the
        # camera, but eliminates tilt).
        theta = rotationMatrixToEulerAngles(np.array(_human_dataset['images'][i]["R"]))
        theta[0] = 0
        theta[1] = 0
        R = eulerAnglesToRotationMatrix(theta)
        
        # Apply the no-tilt rotation matrix
        kpts_3d = np.dot(kpts_3d, R)

        kpts_3d = list(kpts_3d.flatten())
        _human_dataset['annotations'][i]['kpts_3d'] = kpts_3d

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

    # Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                  [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                  [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                  ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                  [0,                     1,      0                   ],
                  [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                  ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                  [math.sin(theta[2]),    math.cos(theta[2]),     0],
                  [0,                     0,                      1]
                  ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R
