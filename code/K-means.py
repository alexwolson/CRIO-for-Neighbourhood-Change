#!/usr/bin/env python
# coding: utf-8
from shared import *


# # K-means clustering first, then linear regression

def initialize_model_kmeans(k, LABEL, f, randnum):
    kmeans_data = readindata_std(LABEL, f)
    kmeans_data = kmeans_data.set_index('trtid')

    kmeans = KMeans(n_clusters=k, init='random', random_state=randnum).fit(
        kmeans_data.iloc[:, 0:f])
    kmeans_model = kmeans.labels_
    kmeans_model = list(map(lambda x: x + 1, kmeans_model))
    kmeans_data = kmeans_data.assign(model=kmeans_model)
    return kmeans_data


def k_means(data, k, f, LABEL):
    kmeans_data = data.copy()
    kmeans_data_x = kmeans_data.iloc[:, 0:f]

    # initialize k linear regression models
    regr_list = []
    for i in range(0, k):
        temp_reg = linear_model.Ridge(
            alpha=0, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto')
        regr_list.append(temp_reg)

    # select data to train each model
    data_temp_list = []
    for i in range(0, k):
        data_temp = kmeans_data.loc[kmeans_data['model'] == (i+1)]
        data_temp_list.append(data_temp)

    # train model k
    for i in range(0, k):
        regr_list[i].fit(data_temp_list[i].iloc[:, 0:f],
                         data_temp_list[i].iloc[:, f:f+1])

    # run prediction for each tract using each model
    train_pred_list = []
    for i in range(0, k):
        pred_temp = regr_list[i].predict(kmeans_data_x)
        train_pred_list.append(pred_temp)

    for i in range(0, k):
        kmeans_data['pred'+str(i+1)] = train_pred_list[i]

    # find mean absolute error per tract
    total_MAEerror = 0
    for _, row in kmeans_data.iterrows():
        label = row[LABEL]
        assigned_model = row['model']
        total_MAEerror = total_MAEerror + \
            math.fabs(float(row['pred'+str(int(assigned_model))])-float(label))

    # find mean square error per tract
    total_MSEerror = 0
    for _, row in kmeans_data.iterrows():
        label = row[LABEL]
        assigned_model = row['model']
        total_MSEerror = total_MSEerror + \
            math.pow(
                (float(row['pred'+str(int(assigned_model))])-float(label)), 2)
    return kmeans_data, regr_list, total_MAEerror/236, total_MSEerror/236


def run_model(k, LABEL, f):
    MAElist = []
    MSElist = []
    regr_list_list = []
    resultlist = []
    for _ in range(0, 50):
        # initialize model
        # random initialization for K-means
        randnum = randint(0, 2**32-1)
        initial_model = initialize_model_kmeans(k, LABEL, f, randnum)

        result, regr_list, MAE, MSE = k_means(initial_model, k, f, LABEL)
        MAElist.append(MAE)
        MSElist.append(MSE)
        regr_list_list.append(regr_list)
        resultlist.append(result)

    return MAElist, MSElist, regr_list_list, resultlist  # ,total_wrong_assignments_list


def collect_result(K, F):
    # k rows, f columns (k = # of clusters, f = # of features)
    MSElist = []
    MAElist = []
    regrlist_list = []
    resultlist = []

    for k in tqdm(range(2, K+1)):
        MSElist_sameCluster = []
        MAElist_sameCluster = []
        regrlist_list_sameCluster = []
        resultlist_sameCluster = []
        allresults_sameCluster = []
        for f in range(2, F+1):
            MAElist_temp, MSElist_temp, regr_list_list_temp, resultlist_temp = run_model(
                k, 'incpc', f)

            allresults_sameCluster.append(resultlist_temp)
            # recording result for k-means as the initialization with lowest MAE
            best_init = MAElist_temp.index(min(MAElist_temp))
            MAElist_sameCluster.append(MAElist_temp[best_init])
            MSElist_sameCluster.append(MSElist_temp[best_init])
            regrlist_list_sameCluster.append(regr_list_list_temp[best_init])
            resultlist_sameCluster.append(resultlist_temp[best_init])
            with open(f'{resultpath}kmeans/rawresults/result_{k}{f}.pickle','wb') as f:
                pickle.dump((allresults_sameCluster,regrlist_list_sameCluster),f)

        MAElist.append(MAElist_sameCluster)
        MSElist.append(MSElist_sameCluster)
        regrlist_list.append(regrlist_list_sameCluster)
        resultlist.append(resultlist_sameCluster)
        
    return MSElist, MAElist, regrlist_list, resultlist  # , testing_MAE, testing_MSE

os.system(f'mkdir {resultpath}kmeans')
os.system(f'mkdir {resultpath}kmeans/rawresults')
kmeansresultlist = display_result(10, 10, 'kmeans', collect_result)
kmeansresult = fixkmeansresultslist(kmeansresultlist)
out = get_silhouette_values(kmeansresult)
print(sorted(out, key=lambda x: x[2]))
with open(resultpath + 'kmeansresultlist.pickle','wb') as f:
    pickle.dump(kmeansresultlist,f)
with open(logpath + 'kmeans.dill', 'wb') as f:
    dill.dump_session(f)
