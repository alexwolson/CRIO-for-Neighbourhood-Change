import warnings, os, re, math, smopy, ast, pycrs, collections, dill, pickle, random
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy.random import randint
from simpledbf import Dbf5
from matplotlib.colors import LogNorm
from statistics import mean
from collections import Counter, defaultdict
from tqdm import tqdm
from copy import deepcopy
from math import pi
from pathlib import Path
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy import stats
random.seed()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

filepath = "../data/"
resultpath = "../results/"
logpath = "../logs/"
LOSS_FUNCTION_SQUARED = False


def myrange(start, end, step):
    while start <= end:
        yield start
        start += step


def getMI(df, LABEL):
    X = df.iloc[:, 0:-1]
    y = df[LABEL].values.flatten()
    mi = mutual_info_regression(X, y)
    mi /= np.max(mi)
    miDF = pd.DataFrame({'feature': X.columns.values})
    miDF = miDF.assign(MI=mi)
    return miDF


def readindata_std(LABEL, f):
    # create data for new york for piecewise regression model
    piecewise_data = pd.read_csv(filepath+"NY_2000_NEW.csv")
    # calculate MI with respect ot the label and rank them
    output = piecewise_data.set_index('trtid')
    miScoredf = getMI(output, LABEL)
    miScoredf = miScoredf.sort_values(by=['MI'], ascending=False)
    # number of features to keep (by highest MI)
    keep_feature = miScoredf[:f]
    kp_feature_list = list(keep_feature['feature'])
    feature_list = ['trtid'] + kp_feature_list
    feature_list.append(LABEL)
    # select index, features, and label from data
    piecewise_data = piecewise_data[feature_list]

    return piecewise_data


def get_coef_models(regr_list, k, f):
    intercept = []
    coef_model = []
    for j in range(0, k):

        # The Intercepts
        intercept.append('cluster '+str(j+1) + ' bias term ' +
                         str(regr_list[j].intercept_))

        # The coefficients
        coef_list = regr_list[j].coef_.tolist()
        # getting all coefficients for one cluster
        flat_list = [item for sublist in coef_list for item in sublist]
        coef_model.append(flat_list)

    feature_list = list(readindata_std('change_incpc', f).iloc[:, 1:f+1].columns)
    Coef = pd.DataFrame({'feature': feature_list})
    for a in range(0, k):
        Coef['Cluster'+str(a+1)] = coef_model[a]

    return intercept, Coef


def cluster_map(result_df, k, f, string):
    # Get census geo data
    df_blocks = gpd.read_file(
        filepath+"geo_export_0e850616-4287-47b8-99de-0bd9ef29b4a3.shp")
    boro_id_dict = {
        'Staten Island': '085',
        'Manhattan': '061',
        'Brooklyn': '047',
        'Queens': '081',
        'Bronx': '005'
    }
    # Construct a tract id to join to the census data
    for key in boro_id_dict:
        df_blocks.loc[df_blocks['boro_name'] == key, 'trtid'] = '36' + \
            boro_id_dict[key] + \
            df_blocks['ct2010'].loc[df_blocks['boro_name'] == key]
    df_blocks['trtid'] = df_blocks['trtid'].astype(np.int64)

    df_to_map = df_blocks.merge(
        result_df, on='trtid', how='inner', suffixes=['blocks_', 'result_'])

    _, ax = plt.subplots(1, figsize=(16, 16))

    df_to_map.plot(ax=ax, column='model', legend=True,
                   cmap=plt.cm.get_cmap('Set3_r', 5))
    os.system(f'mkdir {resultpath}{string}')
    plt.savefig(f'{resultpath}{string}/clustermap{k}{f}.pdf', dpi=300)


def display_result(K, F, string, collect_result, lr=False):
    MSElist, MAElist, _, resultlist = collect_result(K, F)
    with open('resultlist_temp.pickle','wb') as f:
        pickle.dump(resultlist,f)
    # Displaying result for each combination of k and f (# of clusters and # of features)
    for k in range(len(MAElist)):
        for f in range(len(MAElist[k])):
            print(str(k+2 if not lr else 1) + ' cluster, '+str(f+2)+' feature:')
            # MAE training & testing error
            print('MAE(training): ' + str(MAElist[k][f]))
            # MSE training & testing error
            print('MSE(training): ' + str(MSElist[k][f]))
            print('\n')

            # show clustering result on map
            cluster_map(resultlist[k][f], k+2 if not lr else 1, f, string)
    return resultlist


def fixkmeansresultslist(results):
    for i in range(len(results)):
        for j in range(len(results[i])):
            results[i][j]['estimate'] = 0
            for k in range(len(results[i][j])):
                m = results[i][j]['model'].values[k]
                results[i][j]['estimate'] = results[i][j][f'pred{m}']
            results[i][j] = results[i][j].drop(
                [c for c in results[i][j].columns if 'pred' in c], axis=1)
    return results


def get_silhouette_values(results):
    out = []
    for k in range(len(results)):
        for f in range(len(results[k])):
            features_with = results[k][f][results[k][f].columns[:-2]].values
            features_without = results[k][f][results[k][f].columns[:-3]].values
            labels = results[k][f]['model'].values
            sil_with = silhouette_score(features_with, labels, metric="cosine")
            sil_without = silhouette_score(
                features_without, labels, metric="cosine")
            print(f'{k+2} clusters and {f+2} features with: %.2f without: %.2f' %
                  (sil_with, sil_without))
            out.append((k+2, f+2, sil_with, sil_without))
    return out
