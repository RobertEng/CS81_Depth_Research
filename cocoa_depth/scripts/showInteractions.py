############################################
# GENERAL IMPORTS

import sys
import pickle
import json

############################################
# MS COCO SETUP

COCO_IMGS_URLS_FILE = '/home/ubuntu/InteractionsAMTGUI/data/coco_id2url_dict.json'
with open( COCO_IMGS_URLS_FILE ) as f:
    _coco_img_id2url = json.load(f)

COCO_FOLDER = '/mnt/MSCOCO/tools/PythonAPI'
if COCO_FOLDER not in sys.path:
    sys.path.append( COCO_FOLDER )
from pycocotools.coco import COCO

COCO_ANNOTATION_FILE = '/mnt/MSCOCO/annotations/instances_train2014.json'
COCO_IMAGES_FOLDER = '/mnt/MSCOCO/images/train2014'

#coco = COCO( COCO_ANNOTATION_FILE )

############################################
# MONGO CLIENT DATABASE SETUP

from pymongo import MongoClient
_mongo_client = MongoClient()

_mongo_db = _mongo_client.cocoa_5000
_coll1    = _mongo_db.interactions_amt_gui_data

_mongo_coll_2 = _mongo_db.coco_subj_id2coco_img_id
_mongo_coll_3 = _mongo_db.interactions_hit_id2coco_subj_id
_mongo_coll_4 = _mongo_db.coco_subj_id2interactions_hit_id

############################################
# CONSTANTS

OUTPUT_DIR = '/home/ubuntu/InteractionsAMTGUI/plots'
TEMPLATES_DIR = '/home/ubuntu/InteractionsAMTGUI/scripts/templates'

############################################

def main():
    if len(sys.argv) != 3:
        print "Usage: $> python showInteractions.py \'dictionary file\' \'dictionary key\'"
        print "I.e. : $> python showInteractions.py ../hits/dict.pkl _rejected_assignments"
        sys.exit(1)

    dict_path = sys.argv[1]
    dict_key  = sys.argv[2]

    output_name = 'show' + dict_key

    mapster_js = ''

    # completed_cocoa_5000_1951_InteractionHITS_2015-07-14_18-00-10.pkl
    assignments_dict = pickle.load( open(dict_path,'rb') )
    assignments      = assignments_dict[dict_key]

    print dict_path
    print dict_key
    print len(assignments)

    h = open('%s/%s.html'%(OUTPUT_DIR,output_name), 'w')
    h.write('<!DOCTYPE html>\n<html>\n<head>\n<title>Show interactions</title>\n')
    h.write('<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.8.3/jquery.min.js"></script>\n')
    h.write('<script src="/home/ubuntu/both_gui/static/javascript/jquery.imagemapster.js"></script>\n')
    h.write('</head>\n<body>\n')

    for ass in assignments[0:2]:
        worker_exp = ass['worker_exp']
        worker_id  = ass['worker_id']
        ass_id     = ass['assignment_id']
        ass_rt     = ass['response_time']
        
        t = open('%s/assignment_view_template.html'%(TEMPLATES_DIR), 'r')        
        template = ''.join(t.readlines())
        t.close()
        h.write( template%(ass_id, worker_id, worker_exp, (float(ass_rt) / 1000.0)))

        trials     = ass['trials']
        for trial_key in trials:
            trial        = trials[trial_key]

            coco_img_id  = trial['coco_img_id']
            coco_subj_id = trial['coco_subj_id']
            trial_flag   = trial['flag']
            trial_rt     = trial['response_time']
            interactions = trial['interactions']

            mapster_areas = \
                ''.join(_coll1.find({'_coco_img_id':coco_img_id})[0]['_image_mapster_areas'])
            objects_info = \
                json.loads(_coll1.find({'_coco_img_id':coco_img_id})[0]['_obj_info_dict'])
            
            mapster_objects = []
            for key,value in objects_info.items():
                if value['_coco_ann_id'] == coco_subj_id:
                    mapster_subj = key
                
                if value['_coco_ann_id'] in interactions:
                    mapster_objects.append(key)

            #print "+++++++++++++++++++++++++++++++++++++++"
            #print coco_subj_id
            #print interactions
            #print objects_info
            #print mapster_subj
            #print mapster_objects
            #print "======================================="

            mapster_js += write_img_mapster( coco_subj_id, mapster_subj, mapster_objects )
            mapster_js += '\n'
            #print mapster_js

            t = open('%s/interaction_view_template.html'%(TEMPLATES_DIR), 'r')
            template = ''.join(t.readlines())
            t.close()
            h.write( \
                template%( \
                    _coco_img_id2url[str(coco_img_id)], \
                    str(coco_subj_id), str(coco_subj_id), \
                    str(coco_subj_id), mapster_areas, \
                    coco_subj_id, \
                    str(interactions), \
                    str(trial_flag), \
                    (float(trial_rt) / 1000.0) \
                ) \
            )

    h.write('\n</body>\n')
    h.write('<script>\n%s\n</script>\n'%mapster_js)
    h.write('</html>')
    h.close()

def write_img_mapster( coco_subj_id, mapster_subj, mapster_objs ):
    out_string = "$('#%s').mapster({areas: ["%(coco_subj_id)
    out_string += \
    "{key:'%s',staticState:true,fillColor:'6699ff',fillOpacity:0.5,stroke:true,strokeColor:'ff0000',strokeWidth:3}"%(mapster_subj)
    for obj in mapster_objs:
        out_string += \
    ",{key:'%s',staticState:true,fillColor:'8ae65c',fillOpacity:0.7,stroke:true,strokeColor:'ff0000',strokeWidth:3}"%(obj)
    out_string += "],mapKey: 'name'});"

    return out_string

if __name__ == '__main__':
    main()
