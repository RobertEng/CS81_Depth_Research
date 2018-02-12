# use_analysis.py
# uses analysis functions from postprocess_original_utils.py

import json
import numpy as np
from matplotlib import pyplot as plt

from postprocess_original_utils import (load_data, wrongness,
    get_keypoint_comparison_depths, metaperson_comparisons, worker_comparisons)


data = load_data()


from constants import CALTECH_OUTPUT_PATH
data = json.load(open(CALTECH_OUTPUT_PATH, 'r'))
lab = ['nonAMT_607266', 'nonAMT_368102', 'nonAMT_6599', 'nonAMT_687008', 'nonAMT_764039', 'nonAMT_700986']
data = [d for d in data if d['_worker_id'] in lab]
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
'''

# wrongness(data)
metaperson_comparisons(data)
# worker_comparisons(data)
