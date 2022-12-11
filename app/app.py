"""
    This script runs a small Flask app that displays a simple web form for users to insert some input value
    and retrieve predictions.

    Inspired by: https://medium.com/shapeai/deploying-flask-application-with-ml-models-on-aws-ec2-instance-3b9a1cec5e13
"""

from flask import Flask, render_template, request
import numpy as np
from metaflow import Flow
from metaflow import get_metadata, metadata
from flask import Flask,jsonify,request
import json
import pandas as pd
import uuid, time

#### THIS IS GLOBAL, SO OBJECTS LIKE THE MODEL CAN BE RE-USED ACROSS REQUESTS ####

FLOW_NAME = 'Regression' # name of the target class that generated the model
# Set the metadata provider as the src folder in the project,
# which should contains /.metaflow
metadata('../archive/')
# Fetch currently configured metadata provider to check it's local!
print(get_metadata())

example = pd.DataFrame()
example['CNT_CHILDREN'] = [1]
example['AMT_INCOME_TOTAL'] = 200000.0
example['DAYS_BIRTH'] = -10000
example['DAYS_EMPLOYED'] = -3000
example['CNT_FAM_MEMBERS'] = 2.0
example['FLAG_WORK_PHONE'] = 0
example['FLAG_PHONE'] = 0
example['FLAG_EMAIL'] = 0
example['CODE_GENDER'] = 'F'
example['FLAG_OWN_CAR'] = 'N'
example['FLAG_OWN_REALTY'] = 'N'
example['NAME_INCOME_TYPE'] = 'Commercial associate'
example['NAME_EDUCATION_TYPE'] = 'Secondary / secondary special'
example['NAME_FAMILY_STATUS'] = 'Seperated'
example['NAME_HOUSING_TYPE'] = 'Municipal apartment'
example['OCCUPATION_TYPE'] = 'Laborers'

def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():
        if r.successful: 
            return r

# get artifacts from latest run, using Metaflow Client API
latest_run = get_latest_successful_run(FLOW_NAME)
latest_model = latest_run.data.model

# We need to initialise the Flask object to run the flask app 
# By assigning parameters as static folder name,templates folder name
app = Flask(__name__, static_folder='static', template_folder='templates')


@app.route('/predict',methods=['GET'])
def main():
  if request.method=='GET':
        features=['CNT_CHILDREN','AMT_INCOME_TOTAL','DAYS_BIRTH','DAYS_EMPLOYED','CNT_FAM_MEMBERS','FLAG_WORK_PHONE','FLAG_PHONE',
                  'FLAG_EMAIL','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                  'NAME_HOUSING_TYPE','OCCUPATION_TYPE']
        for x in features:
          name=request.args.get(x)
          if type(name) is str:
            example[x]=[float(name)]
          
        #days_employed = request.args.get('DAYS_EMPLOYED')
        #example['DAYS_EMPLOYED']=[float(days_employed)]
        val = latest_model.predict(example)
        id = uuid.uuid4()
        timmme = time.time()
        dic = {'data':{'input':example.values.tolist(),'prediction': val.tolist()}, 'metadata':{'eventID':str(id),'time':timmme}}
        return jsonify(dic)

        
        
    

if __name__=='__main__':
  # Run the Flask app to run the server
  app.run(debug=True)