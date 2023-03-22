#test_app.py
#contains all tests for our ML app
#run with python -m pytest
#expected result: 23 passed, 13 xfailed

from application.models import Entry,UserEntry
from datetime import datetime
import pytest
from flask import json

#Test UserEntry Model (Range test)
@pytest.mark.parametrize("entrylist",[
 [ None,150,None,None,10000000,None,None,10000000000], #integers
 [None, 0.3, None,None,0.00000000001,None,None,0.000005], #floats
 [None, -0.3, None,None,-0.00000000001,None,None,-0.000005], #negatives
 [str('BIGSTRINGBIGSTRING'*50), 1, 'y','n',1,'y','n',1] #strings
])
def test_UserEntryClass(entrylist,capsys):
   with capsys.disabled():
      print(entrylist)
      new_entry = UserEntry( 
      userid = entrylist[0],
      usertype= entrylist[1],
      username = entrylist[2],
      password = entrylist[3])
      assert new_entry.userid == entrylist[0]
      assert new_entry.usertype == entrylist[1]
      assert new_entry.username == entrylist[2]
      assert new_entry.password == entrylist[3]

#Test Entry Model (Range test)
@pytest.mark.parametrize("entrylist",[
 [ None,150,None,None,10000000,None,None,10000000000], #integers
 [None, 0.3, None,None,0.00000000001,None,None,0.000005], #floats
 [None, -0.3, None,None,-0.00000000001,None,None,-0.000005], #negatives
 [str('BIGSTRINGBIGSTRING'*50), 1, 'y','n',1,'y','n',1] #strings
])
def test_EntryClass(entrylist,capsys):
   with capsys.disabled():
      print(entrylist)
      now = datetime.utcnow()
      new_entry = Entry( 
      imgPath = entrylist[0],
      prediction20= entrylist[1],
      label20 = entrylist[2],
      prediction100 = entrylist[3],
      label100 = entrylist[4],
      accuracy100 = entrylist[5],
      predicted_on= now)
      assert new_entry.imgPath == entrylist[0]
      assert new_entry.prediction20 == entrylist[1]
      assert new_entry.label20 == entrylist[2]
      assert new_entry.prediction100 == entrylist[3]
      assert new_entry.label100 == entrylist[4]
      assert new_entry.accuracy100 == entrylist[5]
      assert new_entry.predicted_on == now


#API TESTS
#Getting entry: consistency test
testList=[]
for i in range(20): #test same entry 20 times
   testList.append([11,6,'/upload/forest_1.jpg',	10,	'large_natural_outdoor_scenes', True,	33	,'forest',True])
@pytest.mark.parametrize("entrylist",testList)
def test_getAPI(client, entrylist, capsys):
   with capsys.disabled():
      response = client.get(f'/api/get/{entrylist[0]}')
      #check the result code and headers
      assert response.status_code == 200
      assert response.headers["Content-Type"] =="application/json"
      #check data is correct
      response_body = json.loads(response.get_data(as_text=True))
      assert response_body["id"] == int(entrylist[0])
      assert response_body["userid"] == int(entrylist[1])
      assert response_body["imgPath"] == str(entrylist[2])
      assert response_body["prediction20"] == int(entrylist[3])
      assert response_body["label20"] == str(entrylist[4])
      assert response_body["accuracy20"] == entrylist[5]
      assert response_body["prediction100"] == int(entrylist[6])
      assert response_body["label100"] == str(entrylist[7])
      assert response_body["accuracy100"] == entrylist[8]

#Adding entry: Range test + delete entry afterwards
newEntries=[
 [ 3,'path',2,None,True,50000000,None,True], #integers
 [3, 'path', 10.5,None,True,50.199999,None,False], #floats
 [3, 'path', -20,None,True,-50,None,True], #negatives
 [3,str('BIGSTRINGBIGSTRING'*50), 1, 'y',False,1,'y',False] #strings
]
newEntriesIDs=[]
@pytest.mark.parametrize("entrylist",newEntries)
def test_addAPI(client,entrylist,capsys):
   with capsys.disabled():
      #prepare the data into a dictionary
      data1 = { 
         'userid': entrylist[0],
         'imgPath' : entrylist[1],
         'prediction20': entrylist[2],
         'label20' : entrylist[3],
         'accuracy20' : entrylist[4],
         'prediction100' : entrylist[5],
         'label100' : entrylist[6],
         'accuracy100' : entrylist[7]
      }
      #use client object to post
      #data is converted to json
      #posting content is specified
      response = client.post('/api/add',
      data=json.dumps(data1),
      content_type="application/json",)
      #check the outcome of the action
      assert response.status_code == 200
      assert response.headers["Content-Type"] == "application/json"
      response_body = json.loads(response.get_data(as_text=True))
      id=response_body['id']
      #Delete entry
      response=client.get(f'/api/delete/{id}')
      #check outcome 
      assert response.status_code==200 
      assert response.headers["Content-Type"]=="application/json"
      response.body=json.loads(response.get_data(as_text=True))
      assert response.body["result"]=='ok'
      assert response_body["id"]


#Deleting entry: Expected failure from deleting same entries again
@pytest.mark.xfail(reason="Non-existent record")
@pytest.mark.parametrize("id",[1,1,1,1,1,1])
def test_deleteAPI(client,id,capsys):
   with capsys.disabled():
      response=client.get(f'/api/delete/{id}')
      ret=json.loads(response.get_data(as_text=True))
      #check outcome 
      assert response.status_code==200 
      assert response.headers["Content-Type"]=="application/json"
      response.body=json.loads(response.get_data(as_text=True))
      assert response.body["result"]=='ok'

