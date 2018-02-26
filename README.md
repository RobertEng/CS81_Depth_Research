# CS81 Depth Research

## Lean Correction in Human3.6m Dataset

### The Problem

The issue was raised in https://github.com/una-dinosauria/3d-pose-baseline/issues/37

### Get the Rotation Matrices and Camera Names

On the Human3.6m Downloads page, under the **VISUALIZATION AND LARGE SCALE PREDICTION SOFTWARE** section, download Version 1.1. This contains the code for manipulating the Human3.6m Data. Load the H36M folder into your Matlab workspace. You may need to add `xml_read.m` to your path (`xmlread.m` is in the directory `MATLAB/xml_io_tools_2010_11_05/` for me).

Run `extract_rot_mats.m` (I copy the script into the interactive matlab terminal). This dumps the rotation matrices into `rotationmatrices.mat` and the camera names into `cameranames.mat`.

### Paths
Check the paths in constants.py are correct.

### How to call the correct_lean function
```python
import json
from lean_correction import correct_lean
from constants import HUMAN_ANNOTATION_PATH

with open(HUMAN_ANNOTATION_PATH) as f:
        _human_dataset = json.load(f)
        correct_lean(_human_dataset)
```
