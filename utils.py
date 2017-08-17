"""
set of helping function across whole application
"""
from sklearn.feature_extraction import DictVectorizer
import pickle
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import logging
import os

def findIndeces(l, value):
    """
    find all indeces of value in list l
    :param l:  list of values
    :param value:
    :return:
    """
    result = []
    for i, j in enumerate(l):
        if j == value:
            result.append(i)
    return result

#products of bank
products = ["ind_cco_fin_ult1", "ind_cder_fin_ult1", "ind_cno_fin_ult1", "ind_ctju_fin_ult1",
            "ind_ctma_fin_ult1", "ind_ctop_fin_ult1", "ind_ctpp_fin_ult1", "ind_dela_fin_ult1",
            "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1", "ind_plan_fin_ult1",
            "ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1", "ind_valo_fin_ult1",
            "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1"]



def saveDict(o, name):
    '''
    pickle dict or any object
    :param o:
    :param name:
    :return:
    '''
    with open(name + '.pickle', 'wb') as handle:
        pickle.dump(o, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadDict(name):
    '''
    load pickled dict or any object
    :param name:
    :return:
    '''
    with open(name + '.pickle', 'rb') as handle:
        dict = pickle.load(handle)
        return dict

def addOneHot(dataDf, LOG, train, tresholdOfMinimumOccurence):
    '''
    Create one-hot encoding in data
    :param dataDf: da
    :param LOG: logger
    :param train: if it's train process or not
    :param tresholdOfMinimumOccurence: what is minimum percent of occurance to stay in data for one-hot encoding
    :return:
    '''
    #let's define columns which we want encode
    categoricalColumns =  [
                        'ind_nuevo',       #new customer?
                        'segmento',        #segmenttion
                        'tiprel_1mes',     #Customer relation type at the beginning of the mont
                        'pais_residencia', #place
                        'canal_entrada',   #channel used by the customer to join
                        'sexo',            # sex
                        "indext",          #Foreigner index (S (Yes) or N (No) if the customer's birth country is different than the bank country)
                        "indresi",         #Residence index (S (Yes) or N (No) if the residence country is the same than the bank country)
                        "indrel_1mes",     #Customer type at the beginning of the month
                        "ind_empleado",	   #Employee index
                        "indrel",          #1-(First/Primary), 99 (Primary customer during the month but not at the end of the month
                        "conyuemp",        #Spouse index. 1 if the customer is spouse of an employee
                        "indfall",         #Deceased index. N/S,
                        "nomprov"          #Province
                        ]
    if train:
        usedValuesForCategoricalValues = {}
    else: # test
        usedValuesForCategoricalValues = loadDict("usedValuesForCategoricalValues")
    OTHER_VALUE = "other_value" # value to replace
    for category in categoricalColumns:
        LOG.info('Category processed to one-hot: ' + category)
        if train: #in case of training
            minimumCountToStay = len(dataDf) * tresholdOfMinimumOccurence
            usedValuesForCategoricalValues[category] = set()
            frequencyDict = dataDf[category].value_counts().to_dict()
            for key in frequencyDict: #filter just those values which are enough frequent
                frequencyOfOccurance = frequencyDict[key]
                if frequencyOfOccurance >= minimumCountToStay :
                    usedValuesForCategoricalValues[category].add(key)

        valuesInColumn = set(dataDf[category].unique())
        valuesToReplace = valuesInColumn.difference(usedValuesForCategoricalValues[category])
        if len(valuesToReplace) == 0: #no value to replace
            continue
        else:
            for v in valuesToReplace:
                dataDf[category] =  dataDf[category].replace(to_replace=v, value=OTHER_VALUE)
    if train:
        saveDict(usedValuesForCategoricalValues, "usedValuesForCategoricalValues")
    data, vecData, vec = one_hot_dataframe(dataDf, categoricalColumns)
    result = pd.concat([data, vecData], axis=1)
    result = result.drop(categoricalColumns, axis=1)
    return result


def one_hot_dataframe(data, cols, replace=False):
    '''
    Takes a dataframe and a list of columns that need to be encoded.
    Returns a 3-tuple comprising the data, the vectorized data,
    and the fitted vectorizor.
    :param data:
    :param cols:
    :param replace:
    :return: 3-tuple comprising the data, the vectorized data and the fitted vectorizor
    '''
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)

def get_logger(fname='log.txt'):
    '''
    connect to logger
    :param fname:
    :return:
    '''
    logger = logging.getLogger('my_logger')
    if len(logger.handlers) == 0:
        logger.setLevel(logging.DEBUG)
        path = './log'
        if not os.path.exists(path):
            os.makedirs(path)
        fh = logging.FileHandler('./log/{}'.format(fname))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)-15s %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger

def createSubmision(ids, recommendationsFinal):
    '''
    create submission for kaggle purpose
    :param ids:
    :param recommendationsFinal:
    :return:
    '''
    submisionDF = pd.DataFrame()
    submisionDF["ncodpers"] = ids
    submisionDF["added_products"] = recommendationsFinal
    submisionDF.to_csv("submision.csv",  index=False)