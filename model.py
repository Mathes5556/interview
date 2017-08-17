import pandas as pd
from utils import products, findIndeces
from sklearn.ensemble import RandomForestClassifier
from heapq import nlargest
from metrics import *
from utils import saveDict
from parameters import MIN_PROB_THRESHOLD_FOR_RECOMMENDATION, NUMBER_OF_RECOMMENDATION, \
    RANDOM_FORREST_NUMBER_OF_TREE, RANDOM_FORREST_MAX_FEATURES

DATE_COLUMN_NAME = "fecha_dato" #TODO move
encodingFormat = "ISO-8859-1" #TODO move

def createYVerticeForProduct(df, indexOfProduct):
    """
    Transform boolean vertice description of byuing to 1/0 for given product
    from [[True False False..],[False False False] for product with index 0 => [1, 0] first uswer bought, second not
    :param df: dataframe - Y
    :param indexOfProduct: which product
    :return:  list of 1/0 of given user bought :indexOfProduct:
    """
    y = []
    for boughtVertice in df:
        yIndeces = findIndeces(boughtVertice, True)
        value = 1 if indexOfProduct in yIndeces else 0
        y.append(value)
    return y

def lastPreprocessing(X_TRAIN, Y_TRAIN=None):
    """
    last preprocessing of data before seding them into models - creating from dataframe list of list, delete ambogius columns etc..
    :param X_TRAIN:
    :param Y_TRAIN:
    :return:
    """
    columnsToDropInX = ['ncodpers', 'Unnamed: 0', DATE_COLUMN_NAME, "renta"]  # TODO delete renta
    for c in columnsToDropInX:
        try:
            X_TRAIN = X_TRAIN.drop(c, axis=1)
        except:
            pass
    userExample = X_TRAIN.iloc[0]
    userExample.to_json("user.json")
    X_TRAIN = X_TRAIN.values.tolist()
    if Y_TRAIN is not None:
        Y_TRAIN = Y_TRAIN.values.tolist()
    return X_TRAIN ,Y_TRAIN

def testModels(models, xFileName, yFileName, LOG=None):
    X = pd.read_csv(xFileName, encoding=encodingFormat)
    Y = pd.read_csv(yFileName, encoding="ISO-8859-1")
    # store IDs of user for prediction mapping
    X, Y = lastPreprocessing(X, Y)
    predictionsDf = pd.DataFrame()
    # predict for every product
    for indexOfProduct, product in enumerate(products):
        clf = models[product]
        probs = [list(p)[1] for p in clf.predict_proba(X)]
        predictionsDf[product + "_PROB"] = probs
    recommendedProducts, boughtProducts = [], []
    for i, row in predictionsDf.iterrows():
        probsForUser = list(row)
        largestProbs = nlargest(NUMBER_OF_RECOMMENDATION, list(probsForUser))
        recomndationsForuser = [probsForUser.index(prob) for prob in largestProbs if prob > MIN_PROB_THRESHOLD_FOR_RECOMMENDATION]
        recommendedProducts.append(recomndationsForuser)
    for userDescriptionVertice, userBoughtVertice in zip(X, Y):
        boughtProductsByUser = findIndeces(userBoughtVertice, True)
        boughtProducts.append(boughtProductsByUser)
    return mapk(recommendedProducts, boughtProducts, k=NUMBER_OF_RECOMMENDATION)


def provideRecommendation(models, X, LOG):
    '''
    based on set of models and data provide recommendation
    :param models:
    :param X: dataframe
    :param LOG: logger
    :return: set of recommendation for users
    '''
    LOG.info("Starting of doing recommendation")
    justForOneUser = len(X) == 1 # if we are handling recommnedation just for one user
    idsFromValidation = X['ncodpers'] if not justForOneUser else [None]
    X, _y = lastPreprocessing(X)

    predictionsDf = pd.DataFrame()
    # PREDICT VALIDATION DATA
    for indexOfProduct, product in enumerate(products):
        clf = models[product]
        probs = [list(p)[1] for p in clf.predict_proba(X)]
        predictionsDf[product + "_PROB"] = probs
    recommendedProducts, boughtProducts = [], []
    for i, row in predictionsDf.iterrows():
        probsForUser = list(row)
        largestProbs = nlargest(NUMBER_OF_RECOMMENDATION, list(probsForUser))
        recomndationsForuser = [probsForUser.index(prob) for prob in largestProbs if prob > MIN_PROB_THRESHOLD_FOR_RECOMMENDATION]
        recommendedProducts.append(recomndationsForuser)
    ids, recommendationsFinal = [], []
    for userId, recommendations in zip(idsFromValidation, recommendedProducts):
        productsToRecommend = [products[i] for i in recommendations]
        ids.append(userId)
        # recommendationsFinal.append(" ".join(productsToRecommend)) #kaggle format
        recommendationsFinal.append(productsToRecommend)
    result = {"ids": ids, "recommendations": recommendationsFinal}
    return result

def trainModel(fileNameX, fileNameY, LOG):
    '''
    traing of set of models for eery product
    :param fileNameX:
    :param fileNameY:
    :param LOG:
    :return:
    '''
    X = pd.read_csv(fileNameX, encoding=encodingFormat)
    Y = pd.read_csv(fileNameY, encoding=encodingFormat)
    X, Y = lastPreprocessing(X, Y)
    # here I'm finding indeces of bought product
    yTrainClasses, yTestClasses = [], []
    for ylist in Y:
        indeces = findIndeces(ylist, True)
        yTrainClasses.append(indeces)
    models = {} # dict for storing models for every product

    for indexOfProduct, product in enumerate(products):
        LOG.info('Creating model for product: ' + str(indexOfProduct) + " " + product)
        yForProductTrain = createYVerticeForProduct(Y, indexOfProduct)
        clf = RandomForestClassifier(n_estimators=RANDOM_FORREST_NUMBER_OF_TREE,
                                     max_features=RANDOM_FORREST_MAX_FEATURES
                                     )
        clf.fit(X, yForProductTrain)
        models[product] = clf
    return models
