import glob
import pandas as pd
from utils import products
from utils import addOneHot
from parameters import TRESHOLD_OF_MINIMUM_OCCURENCE

DATE_COLUMN_NAME = "fecha_dato" #TODO move


def afterProcessDataForModel(df, train, LOG):
    '''
    function which provide handling na value and one-hot encoding for selected categorical value
    :param df:
    :param train: if use saved values for one-hot encoding for categorical values
    :param LOG: logger
    :return:
    '''
    df.fillna(value=0)
    df = df.fillna(0)
    df = addOneHot(df, LOG, train=train, tresholdOfMinimumOccurence=TRESHOLD_OF_MINIMUM_OCCURENCE)
    return df

def preparedDataForModel(LOG):
    '''
    Last prepartion of data before sending them to models
    :param LOG:  logger
    :return:
    '''
    monthFiles = glob.glob("trainData/rows*.csv")
    data = None
    for fileName in monthFiles:
        LOG.info('Filname readed: ' + fileName)
        dfLocal = pd.read_csv(fileName, encoding="ISO-8859-1")
        if data is None:
            data = dfLocal
        else:
            data = data.append(dfLocal)

    LOG.info("Preprocess data")
    data = afterProcessDataForModel(data, train=True, LOG=LOG) #TODO this to true!

    Y_COLUMNS = ['boughtNewProductInThisMonth_' + product for product in products]
    COLUMNS_TO_REMOVE = [product for product in products] + Y_COLUMNS + ["Unnamed: 0", "boughtAnyProduct"]
    COLUMNS_TO_REMOVE.append("tiprel_1mes_last") #TODO remove/handle ?
    X_COLUMNS = [column for column in list(data.columns) if column not in COLUMNS_TO_REMOVE]

    trainData = data[data[DATE_COLUMN_NAME] != "2016-05-28'"]
    testData = data[data[DATE_COLUMN_NAME] == "2016-05-28"]


    #save train data
    LOG.info("Creating traing files")
    X_TRAIN = trainData[X_COLUMNS]
    X_TRAIN.to_csv("DATA_TO_MODEL/TRAIN_X.csv")
    Y_TRAIN = trainData[Y_COLUMNS]
    Y_TRAIN.to_csv("DATA_TO_MODEL/TRAIN_Y.csv")
    #save test data
    LOG.info("Creating testing files")
    X_TEST = trainData[X_COLUMNS]
    X_TEST.to_csv("DATA_TO_MODEL/TEST_X.csv")
    Y_TEST = trainData[Y_COLUMNS]
    Y_TEST.to_csv("DATA_TO_MODEL/TEST_Y.csv")

    #TESTING TO KAGGLE FILE (JUNE) - validation dataset
    LOG.info("Creating validation files")
    TEST_DATA_FILE_NAME = "trainData/TEST_DATA.csv"
    validationMonth  = pd.read_csv(TEST_DATA_FILE_NAME, encoding="ISO-8859-1")
    validationMonth = afterProcessDataForModel(validationMonth, train=False, LOG=LOG)

    X_VALIDATION  =  validationMonth[X_COLUMNS]
    X_VALIDATION.to_csv("DATA_TO_MODEL/VALIDATION_X.csv")
    c1 = set(X_VALIDATION.columns)
    c2 = set(X_TRAIN.columns)
    print(c1.difference(c2))

