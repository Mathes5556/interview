import pandas as pd
import numpy as np
import os

def createMonthFiles(LOG):
    '''
    Divide data from big file(over 2GB) to more monthly files and create LAG feature for every row ( if user had product month before)
    :param LOG: logger
    :return:
    '''
    LOG.info('# Creating ./input directory')
    # create input directory
    if not os.path.exists('./input'):
        os.mkdir('./input')

    LOG.info('# Loading train data')
    cols = ["fecha_dato","ncodpers","ind_empleado","pais_residencia","sexo","age","fecha_alta","ind_nuevo","antiguedad","indrel","ult_fec_cli_1t","indrel_1mes","tiprel_1mes","indresi","indext","conyuemp","canal_entrada","indfall","tipodom","cod_prov","nomprov","ind_actividad_cliente","renta","segmento","ind_ahor_fin_ult1","ind_aval_fin_ult1","ind_cco_fin_ult1","ind_cder_fin_ult1","ind_cno_fin_ult1","ind_ctju_fin_ult1","ind_ctma_fin_ult1","ind_ctop_fin_ult1","ind_ctpp_fin_ult1","ind_deco_fin_ult1","ind_deme_fin_ult1","ind_dela_fin_ult1","ind_ecue_fin_ult1","ind_fond_fin_ult1","ind_hip_fin_ult1","ind_plan_fin_ult1","ind_pres_fin_ult1","ind_reca_fin_ult1","ind_tjcr_fin_ult1","ind_valo_fin_ult1","ind_viv_fin_ult1","ind_nomina_ult1","ind_nom_pens_ult1","ind_recibo_ult1"]
    cols_to_remove = ['fecha_alta','ult_fec_cli_1t','tipodom','cod_prov','ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1']
    cols_to_use = [col for col in cols if col not in cols_to_remove]
    data = pd.read_csv('../train_ver2.csv',usecols=cols_to_use, dtype={'indrel_1mes':str, 'conyuemp':str})
    LOG.info('# Fetching date_list, product_list[20]')
    # save all date_list, product_list
    date_list = np.unique(data['fecha_dato']).tolist()


    date_list.append('2016-06-28') #from test data

    product_list = data.columns[data.shape[1]-20:].tolist()
    # 2015-06, 2015-12 ~ 2016-06
    dates = [i for i in range(10,18)] #

    dates.append(5)

    LOG.info('# Inner join with last month')
    ## data 1 : inner join with last month ###
    print(date_list)
    for i in dates:
        LOG.info('# Month : {}'.format(date_list[i]))
        if date_list[i] != '2016-06-28':
            # select current month
            out = data[data.fecha_dato == date_list[i]].reset_index(drop=True)
            if date_list[i] == '2016-05-28':
                print("pre last written!")
                PRELAST_DF = out
            # select last month
            temp = data[data.fecha_dato == date_list[i - 1]][
                product_list + ['ncodpers', 'tiprel_1mes', 'ind_actividad_cliente']].reset_index(drop=True)
            # join
            # log which products user had previous month
            out = out.merge(temp, on='ncodpers', suffixes=('', '_last'))

            # save
            out.to_csv('./input/train_{}.csv'.format(date_list[i]), index=False)
        else:
            #test month
            # select current month
            cols = ["fecha_dato", "ncodpers", "ind_empleado", "pais_residencia", "sexo", "age", "fecha_alta",
                    "ind_nuevo", "antiguedad", "indrel", "ult_fec_cli_1t", "indrel_1mes", "tiprel_1mes", "indresi",
                    "indext", "conyuemp", "canal_entrada", "indfall", "tipodom", "cod_prov", "nomprov",
                    "ind_actividad_cliente", "renta", "segmento"]
            cols_to_remove = ['fecha_alta', 'ult_fec_cli_1t', 'tipodom', 'cod_prov']
            cols_to_use = [col for col in cols if col not in cols_to_remove]

            out = pd.read_csv('../test_ver2.csv', usecols=cols_to_use, dtype={'indrel_1mes': str, 'conyuemp': str})
            for p in product_list:
                out[p] = np.nan

            # select last month
            temp = PRELAST_DF[product_list + ['ncodpers', 'tiprel_1mes', 'ind_actividad_cliente']].reset_index(drop=True)

            # join
            # log which products user had previous month
            out = out.merge(temp, on='ncodpers', suffixes=('', '_last'))
            # save
            print(date_list[i])
            out.to_csv('./input/train_{}.csv'.format(date_list[i]), index=False)
            # # import test (2016-06)
            # cols = ["fecha_dato", "ncodpers", "ind_empleado", "pais_residencia", "sexo", "age", "fecha_alta", "ind_nuevo",
            #         "antiguedad", "indrel", "ult_fec_cli_1t", "indrel_1mes", "tiprel_1mes", "indresi", "indext", "conyuemp",
            #         "canal_entrada", "indfall", "tipodom", "cod_prov", "nomprov", "ind_actividad_cliente", "renta",
            #         "segmento"]
            # cols_to_remove = ['fecha_alta', 'ult_fec_cli_1t', 'tipodom', 'cod_prov']
            # cols_to_use = [col for col in cols if col not in cols_to_remove]
            # out = pd.read_csv('../test_ver2.csv', usecols=cols_to_use,
            #                   dtype={'indrel_1mes': str, 'conyuemp': str}).reset_index(drop=True)
            # # select last month (2016-05)
            # # print("PRE_LAST", date_list[i - 1])
            # temp = PRELAST_DF[product_list + ['ncodpers', 'tiprel_1mes', 'ind_actividad_cliente']].reset_index(drop=True)
            # print(set(PRELAST_DF["fecha_dato"]))
            # print(len(out))
            # print(len(temp))
            # print("columns", len(temp.columns))
            # # join
            # # print(temp.head())
            # # print(list(temp.columns))
            # out = out.merge(temp, on='ncodpers', suffixes=('', '_last'))
            # print("merge")
            # print(len(out), "")
            # print("columns", len(out.columns))
            # # print(list(out.columns))
            # # out.columns = [col + '_last' if col in cols[24:] else col for col in out.columns]
            # # save
            # out.to_csv('./input/train_{}.csv'.format(date_list[i]), index=False)
            # break

