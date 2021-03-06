#######################################################################
# GENERAL PURPOSE MODULES

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import datetime
import time
import random
from operator import itemgetter
import json
import pickle
import sys

#######################################################################
# CONSTANTS

VERBOSE = False
PERSON_CAT_ID = 1
DATASET_MIN_SUBJ_AREA = 1600

#######################################################################
# COCO ANNOTATIONS SETUP

COCO_IMGS_URLS_FILE = '/home/ubuntu/InteractionsAMTGUI/data/coco_id2url_dict.json'
with open( COCO_IMGS_URLS_FILE ) as f:
    _coco_img_id2url = json.load(f)

COCO_FOLDER = '/mnt/MSCOCO/tools/PythonAPI'
if COCO_FOLDER not in sys.path:
    sys.path.append( COCO_FOLDER )
from pycocotools.coco import COCO

COCO_ANNOTATION_FILE = '/mnt/MSCOCO/annotations/instances_train2014.json'
COCO_IMAGES_FOLDER = '/mnt/MSCOCO/images/train2014'

_coco = COCO( COCO_ANNOTATION_FILE )

########################################################################
# MONGO CLIENT DATABASE SETUP

from pymongo import MongoClient
_mongo_client = MongoClient()

_mongo_db = _mongo_client.cocoa_5000_review
_mongo_coll_1 = _mongo_db.interactions_amt_gui_data
_mongo_coll_2 = _mongo_db.coco_subj_id2coco_img_id
_mongo_coll_3 = _mongo_db.interactions_hit_id2coco_subj_id
_mongo_coll_4 = _mongo_db.coco_subj_id2interactions_hit_id

########################################################################
# SELECT IMAGES TO ANNOTATE

DATASET_IMG_ID_LIST = pickle.load(open('/home/ubuntu/InteractionsAMTGUI/hits/cocoa_5000/review/img_id_to_recollect_interactions.pkl','r')) 
NUM_DATASET_IMGS = len(DATASET_IMG_ID_LIST)
NUMBER_HIT_ASSIGNMENTS = 1

_dataset_subj_id_list = []

count = 0
for coco_img_id in DATASET_IMG_ID_LIST:
    count += 1
    if VERBOSE:
        print "[%d]/[%d] -> image id [%d]" %( count, NUM_DATASET_IMGS, coco_img_id )
    else:
        if count == 1 or count == NUM_DATASET_IMGS:
            print "[%d]/[%d] -> image id [%d]" %( count, NUM_DATASET_IMGS, coco_img_id )
    # load all annotations contained in the image coco_img_id
    coco_img_annIds = _coco.getAnnIds( imgIds = coco_img_id, iscrowd = False )
    coco_img_anns = _coco.loadAnns( coco_img_annIds )

    _img_anns_id_list = []
    _img_anns_coord_list = []
    _img_anns_cat_dict = {}

    curr_id = 0
    coco_img_anns = sorted( coco_img_anns, key = itemgetter( 'area' ) ) 
    for ann in coco_img_anns:
        # check if annotation belongs to a person and if so create a special
        # annotation id and a mongo hit that will link this annotation_id with 
        # the whole image annotations 
        if ( ann['category_id'] == PERSON_CAT_ID and ann['area'] >= DATASET_MIN_SUBJ_AREA ):
            tmp_str = 'id_' + str( curr_id )
            #print tmp_str
            _mongo_person_entry = \
                { "_coco_subj_id": ann['id'], 
                  "_coco_img_id": ann['image_id'], 
                  "_subj_img_id": tmp_str}
            res = _mongo_coll_2.insert(_mongo_person_entry)
            _dataset_subj_id_list.append( ann['id'] )

        ann_coords = ann['segmentation']
        for coord in ann_coords:
            # add the current id to the list of annotations in the image 
            # it's an int used to access the annotations in the javascript
            _img_anns_id_list.append( curr_id )
            # add the coordinates of all the segmentations of the annotation
            # note that some annotations may have more than one area
            # not contiguous with each other
            _img_anns_coord_list.append( coord )
        
        tmp_dict = {}
        tmp_dict['_coco_cat_name'] = _coco.loadCats( ann['category_id'] )[0]['name']
        tmp_dict['_coco_ann_id'] = ann['id']
        
        _img_anns_cat_dict['id_' + str(curr_id)] = tmp_dict
        curr_id = curr_id + 1	
    
    _img_mapster_areas_list = []
    for i in range( len( _img_anns_id_list ) ):
        _img_mapster_areas_list.append( \
            '<area shape="poly" name="id_' + \
            str(_img_anns_id_list[i]) + \
            ',all" coords="' + \
            str(_img_anns_coord_list[i]).strip("[]") + \
            '" href="#" />' )
    # create a document in the collection for every coco image 
    # containing all the above annotations
    _mongo_ann_entry = \
        {"_coco_img_id": coco_img_id, 
         "_image_mapster_areas": _img_mapster_areas_list, 
         "_coco_img_src": _coco_img_id2url[ str( coco_img_id ) ], 
         "_obj_info_dict": json.dumps( _img_anns_cat_dict ) }
    # insert the document in the collection
    res = _mongo_coll_1.insert( _mongo_ann_entry )

l = len( _dataset_subj_id_list )
NUMBER_IMAGES_IN_HIT = l

print "_____________________________________________________________"
print "Organizing HITs"
print " - Augmented number of subjects:       [%d]" % l
print " - Number of subjects per HIT:         [%d]" % NUMBER_IMAGES_IN_HIT
print " - Number of annotators per subject:   [%d]" % NUMBER_HIT_ASSIGNMENTS
print " - Total number of HITs needed:        [%d]" %(NUMBER_HIT_ASSIGNMENTS)
print "_____________________________________________________________"

amt_hit_id = 0
for ii in range( 0, NUMBER_HIT_ASSIGNMENTS ):
    random.shuffle( _dataset_subj_id_list )
    
    for jj in range( 0, l, NUMBER_IMAGES_IN_HIT ):
        amt_hit_id = amt_hit_id + 1
        _amt_hit_people_list = _dataset_subj_id_list[jj:jj + NUMBER_IMAGES_IN_HIT]
        
        if VERBOSE:
            print "HITId: [%d] -> coco subjects: [%s]" %( amt_hit_id, str( _amt_hit_people_list ))
        else:
            if amt_hit_id == 1 or amt_hit_id == (NUMBER_HIT_ASSIGNMENTS * (l / NUMBER_IMAGES_IN_HIT)): 
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

