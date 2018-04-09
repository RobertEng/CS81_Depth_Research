#######################################################################
# GENERAL PURPOSE MODULES

import random
import json
import sys

#######################################################################
# CONSTANTS

SEED = 421
random.seed(SEED)

NUMBER_SUBJECTS_IN_HIT = 1
NUMBER_HIT_ASSIGNMENTS = 3

VERBOSE = False

DATASET_MIN_KEYPOINTS = 17

NUM_DATASET_SUBJS = 800

KEYPOINTS_TO_REMOVE = ['left_eye', 'right_eye', 'left_ear', 'right_ear']


#######################################################################
# COCO ANNOTATIONS SETUP

COCO_IMGS_URLS_FILE = ('/home/ubuntu/amt_guis/cocoa_depth/data/'
                       'coco_id2url_dict.json')
with open( COCO_IMGS_URLS_FILE ) as f:
    _coco_img_id2url = json.load(f)

COCO_FOLDER = '/home/ubuntu/datasets/mscoco/tools/PythonAPI'
if COCO_FOLDER not in sys.path:
    sys.path.append( COCO_FOLDER )
from pycocotools.coco import COCO

COCO_ANNOTATION_FILE = '/home/ubuntu/datasets/mscoco/annotations/person_keypoints_train2014.json'
# COCO_IMAGES_FOLDER = '/home/ubuntu/datasets/mscoco/images/train2014'

_coco = COCO( COCO_ANNOTATION_FILE )


########################################################################
# MONGO CLIENT DATABASE SETUP

from pymongo import MongoClient
_mongo_client = MongoClient()

_mongo_db = _mongo_client.cocoa_depth

# TODO: Write a IF EXISTS DROP function
_mongo_db.depth_amt_gui_data.drop()
_mongo_db.keypoint_labels.drop()
_mongo_db.depth_hit_id2coco_subj_id.drop()
_mongo_db.coco_subj_id2depth_hit_id.drop()

_mongo_coll_1 = _mongo_db.depth_amt_gui_data
_mongo_coll_2 = _mongo_db.keypoint_labels
_mongo_coll_3 = _mongo_db.depth_hit_id2coco_subj_id
_mongo_coll_4 = _mongo_db.coco_subj_id2depth_hit_id


########################################################################
# STORE KEYPOINTS AND KEYPOINT LABELS TO BE USED

_labels = _coco.loadCats(1)[0]['keypoints']
_coco_keypoint_labels = _labels[:]

# Remove the undesired keypoints
for _label in KEYPOINTS_TO_REMOVE:
    _coco_keypoint_labels.remove(_label)

# Rename nose to head
_nose_index = _labels.index("nose")
_coco_keypoint_labels[_nose_index] = "head"

# create a single document in the collection and insert it
_mongo_keypoint_entry = { "_keypoint_labels": _coco_keypoint_labels }
res = _mongo_coll_2.insert( _mongo_keypoint_entry )


########################################################################
# SELECT IMAGES TO ANNOTATE

annotations = _coco.loadAnns(_coco.getAnnIds())
# shuffle(annotations)
#annotations = [a for a in annotations if a['num_keypoints'] >= DATASET_MIN_KEYPOINTS][:NUM_DATASET_SUBJS]
annotations = [a for a in annotations if a['num_keypoints'] >= DATASET_MIN_KEYPOINTS]
annotations = [a for a in annotations if all(v == 2 for v in a['keypoints'][2::3])][:NUM_DATASET_SUBJS]
DATASET_IMG_ID_LIST = list(set([a['image_id'] for a in annotations]))


assert(len(annotations) == NUM_DATASET_SUBJS)


_dataset_subj_id_list = []

count = 0
for coco_img_ann in annotations:
    count += 1
    if VERBOSE:
        print "[%d]/[%d] -> subject id [%d]" %( count, NUM_DATASET_SUBJS, coco_img_ann['id'] )
    else:
        if count == 1 or count == NUM_DATASET_SUBJS:
            print "[%d]/[%d] -> subject id [%d]" %( count, NUM_DATASET_SUBJS, coco_img_ann['id'] )

    _dataset_subj_id_list.append( coco_img_ann['id'] )
    
    # Remove the keypoints which we don't want (face keypoints)
    _labels_tmp = _labels[:] 
    for _label in KEYPOINTS_TO_REMOVE:
        #print _label
        _label_index = _labels_tmp.index(_label)
        
        _labels_tmp.remove(_label)
   
        #print _label_index
        #print coco_img_ann['keypoints']
        coco_img_ann['keypoints'][_label_index * 3:_label_index * 3 + 3] = []
        #print coco_img_ann['keypoints']

    # create a document in the collection for every annotation 
    _mongo_ann_entry = \
        {"_coco_img_id": coco_img_ann['image_id'],
         "_coco_subj_id": coco_img_ann['id'],
         "_image_keypoints": coco_img_ann['keypoints'],
         "_keypoints_bbox": coco_img_ann['bbox'],
         "_coco_img_src": _coco_img_id2url[ str( coco_img_ann['image_id'] ) ], 
        }
    # insert the document in the collection
    res = _mongo_coll_1.insert( _mongo_ann_entry )

# randomly add an element to the end of array until full number is divisible by 10
DATASET_SUBJ_ID_LIST = _dataset_subj_id_list
while ( len( DATASET_SUBJ_ID_LIST ) % NUMBER_SUBJECTS_IN_HIT != 0 ):
    DATASET_SUBJ_ID_LIST.append( random.choice( DATASET_SUBJ_ID_LIST ) )

l = len( DATASET_SUBJ_ID_LIST )
print "_____________________________________________________________"
print "Organizing HITs"
print " - Augmented number of subjects:       [%d]" % l
print " - Number of subjects per HIT:         [%d]" % NUMBER_SUBJECTS_IN_HIT
print " - Number of annotators per subject:   [%d]" % NUMBER_HIT_ASSIGNMENTS
print " - Total number of HITs needed:        [%d]" %(NUMBER_HIT_ASSIGNMENTS * (l / NUMBER_SUBJECTS_IN_HIT))
print "_____________________________________________________________"

amt_hit_id = 0
for ii in range( 0, NUMBER_HIT_ASSIGNMENTS ):
    random.shuffle( DATASET_SUBJ_ID_LIST )
    
    for jj in range( 0, l, NUMBER_SUBJECTS_IN_HIT ):
        amt_hit_id = amt_hit_id + 1
        _amt_hit_people_list = DATASET_SUBJ_ID_LIST[jj:jj + NUMBER_SUBJECTS_IN_HIT]
        
        if VERBOSE:
            print "HITId: [%d] -> coco subjects: [%s]" %( amt_hit_id, str( _amt_hit_people_list ))
        else:
            if amt_hit_id == 1 or amt_hit_id == (NUMBER_HIT_ASSIGNMENTS * (l / NUMBER_SUBJECTS_IN_HIT)): 
                print "HITId: [%d] -> coco subjects: [%s]" %( amt_hit_id, str( _amt_hit_people_list ))
        
        # create a document in the collection for every group of coco ids
        _mongo_hit = \
            {"_amt_hit_id": amt_hit_id, 
             "_coco_subjs_ids": _amt_hit_people_list}
        # insert the document in the collection
        res = _mongo_coll_3.insert(_mongo_hit)
        
        for pp in _amt_hit_people_list:
            _mongo_rev_hit = \
            {"_coco_subj_id": pp,
             "_amt_hit_id": amt_hit_id}
            res = _mongo_coll_4.insert( _mongo_rev_hit )

# TODO: I don't believe these asserts will work if the number of subjects in a hit
# is not divisible by total subjects. fix this later. 
assert( _mongo_coll_1.count() == NUM_DATASET_SUBJS )
assert( _mongo_coll_3.count() == NUMBER_HIT_ASSIGNMENTS * (l / NUMBER_SUBJECTS_IN_HIT) )
# TODO: Assert statements on the db collection 4
#assert( _mongo_coll_4.count() == NUMBER_HIT_ASSIGNMENTS * NUM_DATASET_SUBJS )






