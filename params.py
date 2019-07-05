import json
from parameters import Params
def init(train=1):
    #global settings
    if train==1:
        with open("configNetwork.json", 'r') as f:
            settings = json.load(f)
        #print(settings)
    if train==0:
        with open("configNetwork_eval.json", 'r') as f:
            settings = json.load(f)
        print(settings)
    return Params(**settings)
