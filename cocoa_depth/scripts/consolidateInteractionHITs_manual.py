########################################################################
# GENERAL IMPORTS

import os
import sys
import pickle
import json

########################################################################
# MONGO CLIENT DATABASE SETUP

from pymongo import MongoClient
_mongo_client = MongoClient()

_cocoa_5000_db         = _mongo_client.cocoa_5000
_coco_subj_id_2_img_id = _cocoa_5000_db.coco_subj_id2coco_img_id 

_cocoa_interactions_db  = _mongo_client.cocoa_interactions
_cocoa_interactions_tmp = _cocoa_interactions_db.cocoa_interactions_to_consolidate
_cocoa_interactions     = _cocoa_interactions_db.cocoa_interactions

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

def main():
    if len(sys.argv) != 2:
        print "Usage: $> python saveInteractions.py \'review json file\'"
        print "I.e. : $> python saveInteractions.py ../hits/cocoa_5000/review/combinedReviewsInteractionHITs.json"
        sys.exit(1)

    review_file_path = sys.argv[1]
    review_data = json.load(open( review_file_path, "r" ))
    
    add_flag_id_list    = review_data['add_flag_id_list']
    remove_flag_id_list = review_data['remove_flag_id_list']
    discard_img_id_list = review_data['discard_img_id_list']
    print " - Subjects to Add Flag:    [%d]"%len(add_flag_id_list)
    print " - Subjects to Remove Flag: [%d]"%len(remove_flag_id_list)
    print " - Images to discard:       [%d]"%len(discard_img_id_list)
    
    edit_subj_id_list   = review_data['edit_subj_id_list']
    edit_img_id_list    = [_coco_subj_id_2_img_id.find_one({'_coco_subj_id':x})['_coco_img_id'] for x in edit_subj_id_list]
    edit_img_id_list    = list(set(edit_img_id_list))
    print " - Subjects to edit:        [%d]"%len(edit_subj_id_list)
    print " - Images to edit:          [%d]"%len(edit_img_id_list)
    
    tmp = _cocoa_interactions.find()
    cocoa_subjects = [s['coco_subj_id'] for s in tmp if s['flag'] == False]
    print " - Subjects without Flag in COCO-a:        [%d]"%len(cocoa_subjects)
    
    added = 0
    for coco_subj_id in add_flag_id_list:
        added += 1 
        print "--------------------------------------------------------"
        print "Added [%d] Flags"%(added)
        print _cocoa_interactions.find_one({'coco_subj_id':coco_subj_id})
        _cocoa_interactions.update( {'coco_subj_id':coco_subj_id}, { '$set': { 'flag': True } } )
        print _cocoa_interactions.find_one({'coco_subj_id':coco_subj_id})
    
    removed = 0
    for coco_subj_id in remove_flag_id_list:
        removed += 1
        print "--------------------------------------------------------"
        print "Removed [%d] Flags"%(removed)        
        print _cocoa_interactions.find_one({'coco_subj_id':coco_subj_id})
        _cocoa_interactions.update( {'coco_subj_id':coco_subj_id}, { '$set': { 'flag': False } } )
        print _cocoa_interactions.find_one({'coco_subj_id':coco_subj_id})

    tmp = _cocoa_interactions.find()
    cocoa_subjects = [s['coco_subj_id'] for s in tmp if s['flag'] == False]
    print " - Subjects without Flag in COCO-a:        [%d]"%len(cocoa_subjects)

    tmp = _cocoa_interactions.find()
    cocoa_images = [s['coco_img_id'] for s in tmp]
    cocoa_images = list(set(cocoa_images))
    print " - COCO-a images:        [%d]"%len(cocoa_images)
        
    count = 0
    for coco_img_id in discard_img_id_list:
        count += 1
        print "--------------------------------------------------------"
        print "[%d]-[%d]"%(coco_img_id,count)
        print _cocoa_interactions.find({'coco_img_id':coco_img_id}).count()
        _cocoa_interactions.remove( {'coco_img_id':coco_img_id} )
        print _cocoa_interactions.find({'coco_img_id':coco_img_id}).count()

    tmp = _cocoa_interactions.find()
    cocoa_images = [s['coco_img_id'] for s in tmp]
    cocoa_images = list(set(cocoa_images))
    print " - COCO-a images:        [%d]"%len(cocoa_images)

    pickle.dump( edit_img_id_list, open( "../hits/cocoa_5000/review/img_id_to_recollect_interactions.pkl", "wb" ) )
        
if __name__ == '__main__':
    main()


'''
import pickle
_all_assignments = pickle.load( open( "./hits/cocoa_5000/review/cocoa_5000_completed_1_InteractionHITs_combined.pkl", "rb" ) )
_all_reviewed_subj_data = _all_assignments['_good_assignments']['trials']

from pymongo import MongoClient
_mongo_client = MongoClient()
_mongo_client.database_names()
_mongo_cocoa_interactions = _mongo_client.cocoa_interactions
_mongo_cocoa_interactions.collection_names()
all_interactions = _mongo_cocoa_interactions.cocoa_interactions
_mongo_cocoa_interactions_final = _mongo_cocoa_interactions.cocoa_interactions_final

print all_interactions.find().count()

count = 0
inserted = 0
updated = 0

for coco_subj_id in _all_reviewed_subj_data.keys():
    count += 1
    
    print "====[%d]===="%(count)
    print _all_reviewed_subj_data[coco_subj_id]
    print "----------------------------"
    
    # if _all_reviewed_subj_data[coco_subj_id]['flag'] == True:
    #    continue
    
    interaction_entry = all_interactions.find({'coco_subj_id':coco_subj_id})
    if interaction_entry.count() == 0:
        mongo_entry = { \
        'coco_subj_id':_all_reviewed_subj_data[coco_subj_id]['coco_subj_id'],
        'coco_img_id':_all_reviewed_subj_data[coco_subj_id]['coco_img_id'],
        'flag':_all_reviewed_subj_data[coco_subj_id]['flag'],
        'interactions':_all_reviewed_subj_data[coco_subj_id]['interactions'],
        }
        all_interactions.insert( mongo_entry )
        inserted += 1
    else:
        # that subject is present in cocoa_interactions so I update it
        all_interactions.update( {'coco_subj_id':coco_subj_id}, { '$set': { 'flag': _all_reviewed_subj_data[coco_subj_id]['flag'], 'interactions': _all_reviewed_subj_data[coco_subj_id]['interactions'] } } )
        updated += 1
        
    print "----------------------------"

print "updated  [%d]"%updated
print "inserted [%d]"%inserted
print "count    [%d]"%count
print all_interactions.find().count()
'''
