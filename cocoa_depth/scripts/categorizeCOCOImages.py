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

DATASET_MIN_SUBJ_AREA = 1600
OUTPUT_FILE_NAME ='/home/ubuntu/InteractionsAMTGUI/data/mscoco_image_categorization.json'

#######################################################################
# COCO ANNOTATIONS SETUP

COCO_FOLDER = '/mnt/MSCOCO/tools/PythonAPI'
if COCO_FOLDER not in sys.path:
    sys.path.append( COCO_FOLDER )
from pycocotools.coco import COCO

COCO_ANNOTATION_FILE = '/mnt/MSCOCO/annotations/instances_train2014.json'
COCO_IMAGES_FOLDER = '/mnt/MSCOCO/images/train2014'

coco = COCO( COCO_ANNOTATION_FILE )

########################################################################
# SELECT 2500 IMAGES TO ANNOTATE

skipped = 0
no_cat = 0

large_people_id_list = []

PERSON_CAT_ID = coco.getCatIds( catNms = ['person'] )[0]
PERSON_IMG_ID_LIST = coco.getImgIds( catIds = PERSON_CAT_ID )

COCO_SPORTS_CAT_ID  = [x for x in coco.cats if coco.cats[x]['supercategory'] == 'sports']
COCO_OUTDOOR_CAT_ID = [x for x in coco.cats if coco.cats[x]['supercategory'] in ['outdoor', 'vehicle']]
COCO_INDOOR_CAT_ID  = [x for x in coco.cats if coco.cats[x]['supercategory'] in ['indoor', 'appliance', 'furniture', 'kitchen']]

# all possible categorization of an image given scene and number of subjects
crowds_img_id_list                 = []
crowds_sports_img_id_list          = []
crowds_outdoor_img_id_list         = []
crowds_indoor_img_id_list          = []

single_subject_img_id_list         = []
single_subject_sports_img_id_list  = []
single_subject_outdoor_img_id_list = []
single_subject_indoor_img_id_list  = []

groups_img_id_list                 = []
groups_sports_img_id_list          = []
groups_outdoor_img_id_list         = []
groups_indoor_img_id_list          = []

# categorize all the images in MSCOCO containing people (PERSON_IMG_ID_LIST) 

for coco_img_id in PERSON_IMG_ID_LIST:

    annIds = coco.getAnnIds( imgIds = coco_img_id, catIds = PERSON_CAT_ID, iscrowd = False )
    anns = coco.loadAnns( annIds )
    
    # check how many subjects are more than 1600 px 
    large_people = [ x for x in anns if x['area'] >= DATASET_MIN_SUBJ_AREA ]
    num_large_people = len( large_people  )
    if num_large_people == 0:
        # no large people in this image so we don't consider it
        skipped += 1
        continue
        
    # classify images in the categories SPORTS, OUTDOOR OR INDOOR
    # CROWDS, SMALL GROUPS, SINGLE SUBJECTS
    sports_annIds = coco.getAnnIds( imgIds = coco_img_id, catIds=COCO_SPORTS_CAT_ID, iscrowd = False )
    oudoor_annIds = coco.getAnnIds( imgIds = coco_img_id, catIds=COCO_OUTDOOR_CAT_ID, iscrowd = False )
    indoor_annIds = coco.getAnnIds( imgIds = coco_img_id, catIds=COCO_INDOOR_CAT_ID, iscrowd = False )
    
    #print "========================================================="
    #print "sport: [%d][%d][%s]"%(len(sports_annIds),max( len(oudoor_annIds), len(indoor_annIds) ), str(len(sports_annIds) > max( len(oudoor_annIds), len(indoor_annIds) )) )
    #print "out:   [%d][%d][%s]"%(len(oudoor_annIds),max( len(sports_annIds), len(indoor_annIds) ), str(len(oudoor_annIds) > max( len(sports_annIds), len(indoor_annIds) )) )
    #print "in:    [%d][%d][%s]"%(len(indoor_annIds),max( len(oudoor_annIds), len(sports_annIds) ), str(len(indoor_annIds) > max( len(oudoor_annIds), len(sports_annIds) )) )
    #print "========================================================="
    
    if len(sports_annIds) > max( len(oudoor_annIds), len(indoor_annIds) ):
        #print "sport"
        max_sports = True
        max_outdoor = False
        max_indoor = False
    elif len(oudoor_annIds) > max( len(sports_annIds), len(indoor_annIds) ):
        #print "out"
        max_sports = False
        max_outdoor = True
        max_indoor = False
    elif len(indoor_annIds) > max( len(oudoor_annIds), len(sports_annIds) ):
        #print "in"
        max_sports = False
        max_outdoor = False
        max_indoor = True
    else:
        # no clear classification for this image so we don't consider it
        no_cat += 1
        #print "no cat"
        continue
                
    if num_large_people == 1:
        single_subject_img_id_list.append( coco_img_id )
        if max_sports:
            single_subject_sports_img_id_list.append( coco_img_id )
        if max_outdoor:
            single_subject_outdoor_img_id_list.append( coco_img_id )
        if max_indoor:
            single_subject_indoor_img_id_list.append( coco_img_id )
    else:
        if num_large_people <= 4:
            groups_img_id_list.append( coco_img_id )
            if max_sports:
                groups_sports_img_id_list.append( coco_img_id )
            if max_outdoor:
                groups_outdoor_img_id_list.append( coco_img_id )
            if max_indoor:
                groups_indoor_img_id_list.append( coco_img_id )
        else:
            crowds_img_id_list.append( coco_img_id )
            if max_sports:
                crowds_sports_img_id_list.append( coco_img_id )
            if max_outdoor:
                crowds_outdoor_img_id_list.append( coco_img_id )
            if max_indoor:
                crowds_indoor_img_id_list.append( coco_img_id )

    large_people_id_list.extend( [ x['id'] for x in large_people ] )

print "_____________________________________________________________"
print " - Total number of images:      [%d]" % len(PERSON_IMG_ID_LIST)
print " - Images without large people: [%d]" % skipped
print " - Images without category:     [%d]" % no_cat
print " - Remaining Images for cocoa:  [%d]" % (len(PERSON_IMG_ID_LIST) - skipped - no_cat)

print "    - [%d] SINGLE_SUBJECT images"     % len(single_subject_img_id_list)
print "           -> [%d] SPORT"             % len(single_subject_sports_img_id_list)
print "           -> [%d] OUTDOOR"           % len(single_subject_outdoor_img_id_list)
print "           -> [%d] INDOOR"            % len(single_subject_indoor_img_id_list)

print "    - [%d] SMALL GROUPS images"       % len(groups_img_id_list)
print "           -> [%d] SPORT"             % len(groups_sports_img_id_list)
print "           -> [%d] OUTDOOR"           % len(groups_outdoor_img_id_list)
print "           -> [%d] INDOOR"            % len(groups_indoor_img_id_list)

print "    - [%d] CROWD images"              % len(crowds_img_id_list)
print "           -> [%d] SPORT"             % len(crowds_sports_img_id_list)
print "           -> [%d] OUTDOOR"           % len(crowds_outdoor_img_id_list)
print "           -> [%d] INDOOR"            % len(crowds_indoor_img_id_list)

print "    - [%d] SPORT images"              %(len(single_subject_sports_img_id_list) + len(groups_sports_img_id_list) + len(crowds_sports_img_id_list))
print "    - [%d] OUTDOOR images"            %(len(single_subject_outdoor_img_id_list) + len(groups_outdoor_img_id_list) + len(crowds_outdoor_img_id_list))
print "    - [%d] INDOOR images"             %(len(single_subject_indoor_img_id_list) + len(groups_indoor_img_id_list) + len(crowds_indoor_img_id_list))

print " - Total number of subjects:    [%d]" % len(large_people_id_list)
print "_____________________________________________________________"

assert((len(PERSON_IMG_ID_LIST) - skipped - no_cat - \
len(single_subject_img_id_list) - len(groups_img_id_list) - len(crowds_img_id_list)) == 0)
assert((len(PERSON_IMG_ID_LIST) - skipped - no_cat - \
(len(single_subject_sports_img_id_list) + len(groups_sports_img_id_list) + len(crowds_sports_img_id_list)) - \
(len(single_subject_outdoor_img_id_list) + len(groups_outdoor_img_id_list) + len(crowds_outdoor_img_id_list)) - \
(len(single_subject_indoor_img_id_list) + len(groups_indoor_img_id_list) + len(crowds_indoor_img_id_list))) == 0)

coco_imgs_categories_dict = {}

coco_imgs_categories_dict['crowds_sports_img_id_list']          = crowds_sports_img_id_list
coco_imgs_categories_dict['crowds_outdoor_img_id_list']         = crowds_outdoor_img_id_list
coco_imgs_categories_dict['crowds_indoor_img_id_list']          = crowds_indoor_img_id_list

coco_imgs_categories_dict['single_subject_sports_img_id_list']  = single_subject_sports_img_id_list
coco_imgs_categories_dict['single_subject_outdoor_img_id_list'] = single_subject_outdoor_img_id_list
coco_imgs_categories_dict['single_subject_indoor_img_id_list']  = single_subject_indoor_img_id_list

coco_imgs_categories_dict['groups_sports_img_id_list']          = groups_sports_img_id_list
coco_imgs_categories_dict['groups_outdoor_img_id_list']         = groups_outdoor_img_id_list
coco_imgs_categories_dict['groups_indoor_img_id_list']          = groups_indoor_img_id_list

with open(OUTPUT_FILE_NAME, 'w') as outfile:
    json.dump(coco_imgs_categories_dict, outfile)

