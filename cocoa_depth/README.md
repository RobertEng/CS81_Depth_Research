# Setup
The GUI is run using python Flask backend with MongoDB as the database. Routing is done with nginx.


### Setup Mongodb
MongoDB is used to store the data during development and as a backup to AMT.

Install MongoDB from [here](https://docs.mongodb.com/manual/installation/). Create the folder where you want the db to live.
```
mkdir -p /data/db
```

To run the server, run `mongod --dbpath ./data/db`.

### Setup Locally


### Setup Remotely

Check `/etc/nginx/sites-enabled/default` for the configuration of these routes.

# Starting the Server
Start a screen and start the flask server.
```
screen -S cocoa_depth
sudo uwsgi --module pythonServer_v6 --callable app -s /tmp/uwsgi_mturk_cocoa_depth.sock
```

Leave the screen (`Ctrl+A+D`) and change the owner of the server socket.
```
sudo chown -R www-data:www-data /tmp/uwsgi_mturk_cocoa_depth.sock
```

### Querying Mongodb
Here's a couple code snippets to get started in ipython. The first code block here is for querying the Human3.6m data. The second code block is querying the MSCOCO data.
```python
import json
import sys
from pymongo import MongoClient
_mongo_client = MongoClient()

_mongo_db = _mongo_client.cocoa_depth_human36m

_mongo_coll_1 = _mongo_db.depth_amt_gui_data
_mongo_coll_2 = _mongo_db.keypoint_labels
_mongo_coll_3 = _mongo_db.depth_hit_id2human_subj_id
_mongo_coll_4 = _mongo_db.human_subj_id2depth_hit_id

_mongo_coll_5 = _mongo_db.depth_amt_gui_workers
_mongo_coll_6 = _mongo_db.depth_amt_gui_blocked_workers

_mongo_coll_7 = _mongo_db.depth_amt_gui_trials_results
_mongo_coll_8 = _mongo_db.depth_amt_gui_hits_results
```

This is querying the MSCOCO data.
```python
import json
import sys
from pymongo import MongoClient
_mongo_client = MongoClient()

_mongo_db = _mongo_client.cocoa_depth

_mongo_coll_1 = _mongo_db.depth_amt_gui_data
_mongo_coll_2 = _mongo_db.keypoint_labels
_mongo_coll_3 = _mongo_db.depth_hit_id2coco_subj_id
_mongo_coll_4 = _mongo_db.coco_subj_id2depth_hit_id

_mongo_coll_5 = _mongo_db.depth_amt_gui_workers
_mongo_coll_6 = _mongo_db.depth_amt_gui_blocked_workers

_mongo_coll_7 = _mongo_db.depth_amt_gui_trials_results
_mongo_coll_8 = _mongo_db.depth_amt_gui_hits_results
```

Useful code tidbits.
```python
# example of a drop
# _mongo_coll_7.drop()

_mongo_coll_8.find_one()

cursor = _mongo_coll_8.find({})
for document in cursor:
    # print document
    if document['_hit_comment'] != "":
        # print document['_hit_id']
        print document['_hit_comment']
```

# Amazon Mechanical Turk
Click on My Account. The menu shows how much money is left in this account i.e. $521.11.

## AMT Sandbox
Always test on sandbox first before deploying. Go to https://requestersandbox.mturk.com/ and sign in as a requester.
Click on Manage menu. Click on Manage Hits Individually. This is where you see each hit individually.

## Mturk Functions
All the important information is in the function getHITType in mturk_api. To create hits, run the mturk_api function createHITS() in ipython.
```python
from mturk_api import mturk_depth_api
mturk_depth_api.createHITs()

mturk_depth_api.deleteAllHits()
```

## Deploying to AMT
When deploying checklist:
* change MAX_HITS in mturk_depth_api to 800 or 100 or whatever.
* Change the url setup for mechanical turk.
* Change the external_url in createHITs to appropriate route.
* Check reward price, title, and description, keywords. etc. 17 cents is too much. Can you Guess the Closest Thing in the Image? as a title. NUMBER_ASSIGNEMTNS would be 3.
* Uncomment the qualifications. Qualifications is for people, keep it to US or Canada. include a numberhitsapprovedrequirement to like 100. also include number of hits approve to 98%.
* Clear the mongodb coll_7 and coll_8.
* Change the url in the GUI html form between sandbox and normal. 
* Make sure to reload the mturk_api into ipython after making changes.

Process to deploy:
* Always test on sandbox first.
* Sign in as requester. Sign in as mronchi@caltech.edu and password blah. Go to manage, manage hits individually.
* from within the server, look into mturk_api file. Do the deploy checklist from above.
* Navigate to the directory of the app or whatever. Open up an ipython and run the api stuff like createhits.
* Check on the sandbox it worked.
* Then delete all the sandbox hits after you make sure it works.

When they are deployed, you have to process and manage them. 
* getReviewableHITs is if all 3 assignments have been completed.
* getReviewableAssignments works if you complete a assignments.
* mturk_depth_api processAssignments() and get all that data.
* For each assignment you pull, there is an assignment id.
* If you want to check stuff in the mturk api, go to boto mturk api. Some cloudhackers stuff.

Go to /etc/nginx/sites_enabled/default and do something?  

Timeline
* Deployed Tuesday 11:13pm, $0.04. Restarted with $0.05 around midnight.
* Deleted and restarted with $0.06 at 10am on wedenesday.
* Deleted and restarted with 100 HITS at $0.07 at 11am.
* No results in anything. Redid everything with https. Redeployed 100 hits on Wednesday 5pm. ~32 hits were done of 300 from 5-6pm.
* 6-9pm, seems to be done with 160 hits of 300.
* 430 assignments at 2:30pm. 438 at 3:00. THAT RATE IS SO SLOW??? O.o

Running mturk_depth_api processAssignments() at friday 4am. generating at /home/ubuntu/amt_guis/cocoa_depth/hits/coco/cocoa_test_completed_658_DepthHITS_2017-06-09_11-06-29.pkl
Running mturk_depth_api_human processAssignments() at friday 4am. generating at /home/ubuntu/amt_guis/cocoa_depth/hits/human/cocoa_test_completed_50_DepthHITS_2017-06-09_11-12-18.pkl





