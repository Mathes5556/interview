from createMonthFiles import createMonthFiles
from utils import get_logger
from processMonthFiles import processMonthFiles
from  model import *
from prepareData import preparedDataForModel
from utils import saveDict, loadDict, createSubmision

MODEL_NAME_ON_DISK = 'models'
encodingFormat = "ISO-8859-1" #TODO move

def wholeTraingFlow(LOG):
    """
    Apply whole traine flow (preprocessing and traing of models)
    :return: dict of models - each bank product  has own model
    """
    LOG.info('=' * 50)
    LOG.info("Starting of traing flow")
    createDataFromScratch = False # if we want to re-create all data again
    if createDataFromScratch:
        createMonthFiles(LOG)
        processMonthFiles(LOG)
        preparedDataForModel(LOG)
    trainAgain = False  # if we want to re-train models
    if trainAgain:
        models = trainModel("DATA_TO_MODEL/TRAIN_X.csv", "DATA_TO_MODEL/TRAIN_Y.csv", LOG)
        saveDict(models, MODEL_NAME_ON_DISK)
        LOG.info("Models were pickled")
    else:
        try: # if exist already models
            models = loadDict(MODEL_NAME_ON_DISK)
            LOG.info("Loading of models was succesfull!")
        except:
            LOG.info("Loading of models wasn't succesfull - will be retrained!  ")
            models = trainModel("DATA_TO_MODEL/TRAIN_X.csv", "DATA_TO_MODEL/TRAIN_Y.csv", LOG)
            saveDict(models, MODEL_NAME_ON_DISK)
            
    return models
