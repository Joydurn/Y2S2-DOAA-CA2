import pytest
import requests
import base64
import json
from tensorflow.keras.datasets.cifar100 import load_data
import numpy as np
#load CIFAR100 dataset
(_, _), (x_test, y_test100) = load_data(label_mode='fine') #cifar 100 images and labels
(_, _), (_, y_test20) = load_data(label_mode='coarse') #cifar20 labels

#server URLs
url1 = 'http://models_server:8501/v1/models/cifar/versions/1:predict' 
url2 = 'http://models_server:8501/v1/models/cifar/versions/2:predict' 

def make_prediction100(instances):
 data = json.dumps({"signature_name": "serving_default",
 "instances": instances.tolist()}) 
 headers = {"content-type": "application/json"}
 json_response = requests.post(url1, data=data, headers=headers)
 predictions = json.loads(json_response.text)['predictions']
 return predictions

def test_prediction100():
 predictions = make_prediction100(x_test[0:100]) 
 accurateCount=0
 for i, pred in enumerate(predictions):
   print(y_test100[i],np.argmax(pred))
   if y_test100[i] == np.argmax(pred): #if accurate 
      accurateCount+=1
 assert accurateCount>=20 #at least 20% accuracy


def make_prediction20(instances):
 data = json.dumps({"signature_name": "serving_default",
 "instances": instances.tolist()}) 
 headers = {"content-type": "application/json"}
 json_response = requests.post(url2, data=data, headers=headers)
 predictions = json.loads(json_response.text)['predictions']
 return predictions

def test_prediction20():
 predictions = make_prediction20(x_test[0:100]) 
 accurateCount=0
 for i, pred in enumerate(predictions):
   if y_test20[i] == np.argmax(pred): #if accurate 
      accurateCount+=1
 assert accurateCount>=40 #at least 40% accuracy