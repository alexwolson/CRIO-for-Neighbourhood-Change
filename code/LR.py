#!/usr/bin/env python
# coding: utf-8
from shared import *
# # Piecewise Linear Regression Model


def initialize_model_random(k, n, LABEL, f):
    # n = number of tracts

    #read in data
    piecewise_data = readindata_std(LABEL, f)

    # randomly generate an array of numbers between 1(inclusive) and k (exclusive)
    initial = randint(1, k+1, n)

    # add the model ID to a column called "model" in the dataset
    piecewise_data = piecewise_data.assign(model=initial)
    piecewise_data = piecewise_data.set_index('trtid')
    return piecewise_data


# re-assign single tract at a time, re-assign the one with greatest improvement in terms of error
# piecewise_data input into the function is already initialized
# data is dataframe of training data
# k: number of models
# Lambda: value of regularizer
# f: number of features used
def algo_model2(data, k, f, LABEL):
    piecewise_data = data.copy()

    # stop the algorithm when there is no better linear regressions
    terminate = 0
    # getting only the feature columns
    piecewise_data_x = piecewise_data.iloc[:, 0:f]

    iteration = 0
    while terminate == 0:

        # initialize k linear regression models
        regr_list = []
        for i in range(0, k):
            temp_reg = linear_model.Ridge(
                alpha=0, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto')
            regr_list.append(temp_reg)

        # select data to train each model
        data_temp_list = []
        for i in range(0, k):
            data_temp = piecewise_data.loc[piecewise_data['model'] == (i+1)]
            data_temp_list.append(data_temp)
            # print(str(i+1)+str(data_temp.shape))

        # train model k
        for i in range(0, k):
            regr_list[i].fit(data_temp_list[i].iloc[:, 0:f],
                             data_temp_list[i].iloc[:, f:f+1])

        # run prediction for each tract using each model
        train_pred_list = []
        for i in range(0, k):
            pred_temp = regr_list[i].predict(piecewise_data_x)
            # print(pred_temp)
            train_pred_list.append(pred_temp)

        # for first iteration, assign prediction for each model to a new column in dataset
        if iteration == 0:
            for i in range(0, k):
                piecewise_data['pred'+str(i+1)] = train_pred_list[i]

        else:
            # for iterations after the first one, update the prediction columns for each model
            for i in range(0, k):
                piecewise_data = piecewise_data.drop(columns=['pred'+str(i+1)])
                piecewise_data['pred'+str(i+1)] = train_pred_list[i]

        max_improvement = 0
        change_index = 0
        change_model = 0

        # find model with lowest prediction error for each tract
        for index, row in piecewise_data.iterrows():
            true = row[LABEL]
            old_model = row['model']

            # getting prediction for each model for this tract
            compare_list = []
            for i in range(0, k):
                temp = math.pow(math.fabs(float(row['pred'+str(i+1)])-float(true)),2)
                # temp=(float(row['pred'+str(i+1)])-float(true))*(float(row['pred'+str(i+1)])-float(true))
                compare_list.append(temp)

            # getting model with lowest absolute prediction error
            new_model = compare_list.index(min(compare_list))+1
            improvement = math.fabs(
                math.pow(float(row['pred'+str(int(old_model))])-float(true),2)) - min(compare_list)

            # check if improvement for this tract is greater than max improvement
            if improvement > max_improvement:
                change_index = index
                max_improvement = improvement
                change_model = new_model

        # if there is improvement, re-assign tract with greatest improvement
        if max_improvement != 0:
            piecewise_data.at[change_index, 'model'] = change_model
            terminate = 0
        # if there is no improvement, do not assign any tract
        else:
            terminate = 1
        iteration = iteration+1
        # print("iteration"+str(iteration))

    # find mean absolute error per tract
    total_MAEerror = 0
    for index, row in piecewise_data.iterrows():
        label = row[LABEL]
        assigned_model = row['model']
        total_MAEerror = total_MAEerror + \
            math.fabs(float(row['pred'+str(int(assigned_model))])-float(label))

    # find mean square error per tract
    total_MSEerror = 0
    for index, row in piecewise_data.iterrows():
        label = row[LABEL]
        assigned_model = row['model']
        total_MSEerror = total_MSEerror + \
            math.pow(
                (float(row['pred'+str(int(assigned_model))])-float(label)), 2)
    return piecewise_data, regr_list, iteration, total_MAEerror/236, total_MSEerror/236


def run_model(k, LABEL, f):
    MAElist = []
    MSElist = []
    regr_list_list = []
    resultlist = []

    for _ in range(0, 1):
        # initialize model
        initial_model = initialize_model_random(k, 236, LABEL, f)

        result, regr_list, _, MAE, MSE = algo_model2(
            initial_model, k, f, LABEL)
        MAElist.append(MAE)
        MSElist.append(MSE)
        regr_list_list.append(regr_list)
        resultlist.append(result)

    # ,total_wrong_assignments_list
    return MAElist, MSElist, regr_list_list, resultlist

def collect_result(K, F):
    # k rows, f columns (k = # of clusters, f = # of features)
    MSElist = []
    MAElist = []
    regrlist_list = []
    resultlist = []

    for k in range(1, K+1):
        MSElist_sameCluster = []
        MAElist_sameCluster = []
        regrlist_list_sameCluster = []
        resultlist_sameCluster = []
        for f in range(2, F+1):
            MAElist_temp, MSElist_temp, regr_list_list_temp, resultlist_temp = run_model(
                k, 'incpc', f)

            # recording result for k-means as the initialization with lowest MAE
            best_init = MAElist_temp.index(min(MAElist_temp))
            MAElist_sameCluster.append(MAElist_temp[best_init])
            MSElist_sameCluster.append(MSElist_temp[best_init])
            regrlist_list_sameCluster.append(regr_list_list_temp[best_init])
            resultlist_sameCluster.append(resultlist_temp[best_init])

        MAElist.append(MAElist_sameCluster)
        MSElist.append(MSElist_sameCluster)
        regrlist_list.append(regrlist_list_sameCluster)
        resultlist.append(resultlist_sameCluster)

    return MSElist, MAElist, regrlist_list, resultlist


results = display_result(1, 5, 'lr', collect_result, True)
results = fixkmeansresultslist(results)
with open(logpath + 'lr.dill', 'wb') as f:
    dill.dump_session(f)
