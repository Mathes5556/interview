import pandas as pd
from utils import products #all products from data-set
import glob
import numpy as np


BOUGHT_ANY_PRODUCT_COLUMN_NAME = "boughtAnyProduct"
USER_ID_COLUMN_NAME = "ncodpers"
DATE_COLUMN_NAME = "fecha_dato"


def processMonthFiles(LOG):
    '''
    In begging of this function we have X files, each of the descripe month
    This function is resposnible for filter those data and keep just rows where users bought any product
    :param LOG:  logger
    :return:
    '''

    monthFiles = glob.glob("input/*.csv")
    sorted(monthFiles)
    # first prepare last month as a dataset for testing/eval
    lastMonthFileName = monthFiles[-1]
    lastMonthDf = pd.read_csv(lastMonthFileName, encoding="ISO-8859-1")
    lastMonthDf[BOUGHT_ANY_PRODUCT_COLUMN_NAME] = np.nan
    lastMonthDf.to_csv("trainData/TEST_DATA.csv")
    monthCounting = 0
    for lastMonthFileName, curentMonthFileName in zip(monthFiles, monthFiles[1:]):
        monthCounting += 1
        lastMonthDf = pd.read_csv(lastMonthFileName, encoding="ISO-8859-1")
        curentMonthDf = pd.read_csv(curentMonthFileName, encoding="ISO-8859-1")
        twoMonthTogether = lastMonthDf.append(curentMonthDf)
        #let's sort dateframe by userId and date and after-that we can use shift function for LAG (shift in pandas)
        twoMonthTogether = twoMonthTogether.sort([USER_ID_COLUMN_NAME, DATE_COLUMN_NAME])
        t = twoMonthTogether[twoMonthTogether.ncodpers == 15901]
        t.to_csv("t" + str(monthCounting)  +  ".csv")
        # products = products[1:3]
        for product in products:
            # calculate lag-1 feature (change in product status) - shift(periods=1) see the previous row

            isSameUserAsMonthBefore = twoMonthTogether[USER_ID_COLUMN_NAME] == twoMonthTogether.shift(periods=1)[USER_ID_COLUMN_NAME]

            hasNotPreviousMonthProduct = twoMonthTogether.shift()[product] == 0

            hasInThisMonthProduct = twoMonthTogether[product] == 1

            boughtNewProductInThisMonth = isSameUserAsMonthBefore & hasNotPreviousMonthProduct & hasInThisMonthProduct

            twoMonthTogether['boughtNewProductInThisMonth_' + product] = boughtNewProductInThisMonth

        #filter just those which bought any product in given month
        boughtAnyProduct = twoMonthTogether[['boughtNewProductInThisMonth_' + product for product in products]].sum(axis=1)

        twoMonthTogether['boughtAnyProduct'] = boughtAnyProduct

        rowsWhichBoughtSomething = twoMonthTogether[twoMonthTogether['boughtAnyProduct'] > 0]
        rowsWhichBoughtSomething.to_csv("trainData/rowsWhichBoughtSomething" + str(monthCounting) + " .csv")