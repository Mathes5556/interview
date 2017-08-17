from flask import Flask, request
from traingFlow import  wholeTraingFlow
from utils import get_logger
from model import provideRecommendation
import pandas as pd
from model import provideRecommendation

app = Flask(__name__)


global models # well, global variables are bad, but this is for storing trained models during running of server
models = []

LOG = get_logger('RECOMENDATION_EXPONEA_LOG.log') #initialize of logger

@app.route('/')
def about():
    '''
    Just for ensure server works
    :return:
    '''
    return 'Hello, World!'

@app.route('/train')
def train():
    '''
    Whole process of training applied on history data
    :return:
    '''
    global models
    models = wholeTraingFlow(LOG)
    LOG.info("Models for products were sucesfully created!")
    return "model was just trained!"

@app.route('/recommendationForUser/', methods=['POST'])
def recommendUser():
    '''
    based on POST user data get recommendation
    :return:  list of products which model recommend for given user
    '''
    global models
    if len(models) == 0:
        return "Please train models first!"
    userJson = request.get_json()
    userJsonReadyToDF = {}
    for k in userJson:
        userJsonReadyToDF[k] = [userJson[k]]
    result = provideRecommendation(models, pd.DataFrame.from_dict(userJsonReadyToDF), LOG )
    return "recomended products: " + str(result["recommendations"][0])

if __name__ == "__main__":
    app.run()