#!/usr/bin/env python
# coding: utf-8

import dill
from gurobipy import *
from shared import *
from milpshared import *

def MIP_model_BigM(LABEL, numTracts, numModels, numFeatures, runtimelimit, M_val):
    # read in feature value and label value from dataframe
    DF = readindata_std(LABEL, numFeatures)
    df = DF.copy()

    M = M_val

    # feature data
    # create feature value list Xij
    X_val = df.iloc[:, 1:numFeatures+1].values.tolist()
    Y = df.iloc[:, -1].tolist()  # create label value list Yi

    model = Model()

# Basically, I've just dropped lines with a Z -- since the weight regularizer was removed, this part is no longer used (should not affect optimization, but good to remove it just to be safe).  -Scott

    # Add variables
    X = {}
    E = {}
    W = {}
    B = {}
    C = {}
    for i in range(numTracts):
        for j in range(numFeatures):
            X[(i, j)] = X_val[i][j]

    for i in range(numTracts):
        for k in range(numModels):
            E[(i, k)] = model.addVar(
                lb=0, vtype=GRB.CONTINUOUS, name="E%d,%d" % (i, k))

    for j in range(numFeatures):
        for k in range(numModels):
            W[(j, k)] = model.addVar(vtype=GRB.CONTINUOUS, name="W%d,%d" % (j, k))

    for k in range(numModels):
        B[k] = model.addVar(vtype=GRB.CONTINUOUS, name="B%d" % k)

    for i in range(numTracts):
        for k in range(numModels):
            C[(i, k)] = model.addVar(vtype=GRB.BINARY, name="C%d,%d" % (i, k))

    model.update()

    # Add constraints
    for i in range(numTracts):
        model.addConstr(quicksum(C[(i, k)] for k in range(numModels)) == 1)

    for i in range(numTracts):
        for k in range(numModels):
            model.addConstr(quicksum(W[(j, k)]*X[(i, j)] for j in range(
                numFeatures)) + B[k] - Y[i] - E[(i, k)] <= M*(1-C[(i, k)]))

    for i in range(numTracts):
        for k in range(numModels):
            model.addConstr(quicksum(-W[(j, k)]*X[(i, j)] for j in range(
                numFeatures)) - B[k] + Y[i] - E[(i, k)] <= M*(1-C[(i, k)]))

 # set objective
    model.setObjective( quicksum( quicksum( E[(i,k)] for i in range(numTracts)) for k in range(numModels)))
    model.Params.timeLimit = runtimelimit  # 12 hours
#     model.Params.LogFile = filepath+"MIP_bigM_real_log_m"+str(numModels)+"_f"+str(numFeatures)
    model.optimize()
#     model.write(filepath+"MIP_bigM_real_m"+str(numModels)+"_f"+str(numFeatures)+".sol")

    df = pd.DataFrame(columns=['Dec_Var', 'Val'])
    for v in model.getVars():
        df = df.append({'Dec_Var': v.varName, 'Val': v.x}, ignore_index=True)

    error_list = []
    error_list = [x.X for x in model.getVars() if x.VarName.find('E') != -1]

#     for b in myrange(0,numTracts*numModels-1,numModel):
#         if model_list_raw[b]==1:
#             mo

    bias_list = [x.X for x in model.getVars() if x.VarName.find('B') != -1]

    coef_list = [x.X for x in model.getVars() if x.VarName.find('W') != -1]

    MAE = 0
    for a in range(0, numTracts*numModels):
        MAE = MAE + error_list[a]
    MAE = MAE/numTracts

    MSE = 0
    for a in range(0, numTracts*numModels):
        MSE = MSE + math.pow(error_list[a], 2)
    MSE = MSE/numTracts

#     weights_df = df.iloc[236*numModels:(181*numModels+numFeatures*numModels),:]
#     intercept_df = df.iloc[(236*numModels+numFeatures*numModels):(181*numModels+numFeatures*numModels+numModels),:]
#     model_df = df.iloc[(236*numModels+numFeatures*numModels+numModels):(181*numModels+numFeatures*numModels+numModels+181*numModels),:]

    # return df, error, weights_df, intercept_df, model_df,model.MIPGap*100
    return df, MAE, MSE, bias_list, coef_list, model.MIPGap*100


def collect_result(K, F):
    # k rows, f columns (k = # of clusters, f = # of features)
    MSElist = []
    MAElist = []
    Coeflist = []
    Biaslist = []
    resultlist = []

    for k in tqdm(range(2, K+1)):
        MSElist_sameCluster = []
        MAElist_sameCluster = []
        Coeflist_sameCluster = []
        Biaslist_sameCluster = []
        resultlist_sameCluster = []

        for f in range(2, F+1):

            # run the MILP model
            M_val = pairwise_distance(236, 'incpc', f,k)
            result, MAE, MSE, bias_list, coef_list, _ = MIP_model_BigM(
                'incpc', 236, k, f, 3600, M_val)

            # recording training MAE, MSE for MILP
            MAElist_sameCluster.append(MAE)
            MSElist_sameCluster.append(MSE)

            # recording Bias term for MILP
            Biaslist_sameCluster.append(bias_list)

            # recording regression coefficients for MILP
            coef_model = []
            for a in range(0, k):
                # getting all coefficients for one cluster
                flat_list = []
                for b in range(0, f):
                    flat_list.append(coef_list[a+b*k])
                coef_model.append(flat_list)

            feature_list = list(readindata_std(
                'incpc', f).iloc[:, 1:f+1].columns)
            Coef = pd.DataFrame({'feature': feature_list})
            for c in range(0, k):
                Coef['Cluster'+str(c+1)] = coef_model[c]
            Coeflist_sameCluster.append(Coef)

            # convert result into dataframe, each tract pair with its cluster assignment
            result_df = result.copy()
            trtid_df = readindata_std('incpc', f)
            result_df = result_df[result_df['Dec_Var'].str.contains("C")]
            result_df = result_df[result_df['Val'] > 0.9]
            model_list = []
            for _, row in result_df.iterrows():
                assigned_label_text = row['Dec_Var']
                assigned_label = int(assigned_label_text[-1])+1
                model_list.append(assigned_label)
            trtid_df = trtid_df.assign(model=model_list)
            trtid_df = trtid_df.set_index('trtid')
            resultlist_sameCluster.append(trtid_df)

            bias_List = []
            for h in range(0, k):
                bias_List.append([bias_list[h]])
            with open(f'{resultpath}milp/rawresults/result_{k}{f}.pickle','wb') as f:
                pickle.dump((resultlist_sameCluster,(Coeflist_sameCluster, Biaslist_sameCluster)),f)


        # recording result for k-means as the initialization with lowest MAE
        MAElist.append(MAElist_sameCluster)
        MSElist.append(MSElist_sameCluster)
        Coeflist.append(Coeflist_sameCluster)
        Biaslist.append(Biaslist_sameCluster)
        resultlist.append(resultlist_sameCluster)

    return MSElist, MAElist, (Coeflist, Biaslist), resultlist


MILP_result = display_result(5, 5, 'milp', collect_result)


def overlap(K, F, MILP_result_df):
    MILP_result = MILP_result_df.copy()
    with open(resultpath + 'kmeansresultlist.pickle','rb') as f:
        Kmeans_result_df = pickle.load(f)
    Kmeans_result = Kmeans_result_df.copy()
    # for each combination of # of clusters & # of features
    kmeans_pairID_list = []
    kmeans_intersection_list = []
    Jaccard_AB_list = []
    Jaccard_A_list = []
    Jaccard_B_list = []
    Jaccard_index_sum_list = []
    Jaccard_index_min_list = []
    for k in range(2, K+1):
        for f in range(2, F+1):
            kmeans_cluster = []
            MILP_cluster = []

            # store trtid within each cluster for kmeans and MILP seperately
            for a in range(0, k):
                Kmeans_result[k-2][f-2] = Kmeans_result[k-2][f-2].reset_index()
                temp_kmeans = Kmeans_result[k-2][f -
                                                 2].loc[Kmeans_result[k-2][f-2]['model'] == a+1]
                kmeans_cluster.append(
                    temp_kmeans['trtid'].values.flatten().tolist())

                MILP_result[k-2][f-2] = MILP_result[k-2][f-2].reset_index()
                temp_MILP = MILP_result[k-2][f -
                                             2].loc[MILP_result[k-2][f-2]['model'] == a+1]
                MILP_cluster.append(
                    temp_MILP['trtid'].values.flatten().tolist())

                Kmeans_result[k-2][f-2] = Kmeans_result[k -
                                                        2][f-2].set_index('trtid')
                MILP_result[k-2][f-2] = MILP_result[k -
                                                    2][f-2].set_index('trtid')

            # pair kmeans and MILP cluster to maximize interseted elements
            kmeans_pairID = []
            kmeans_intersection = []
            Jaccard_AB = []
            Jaccard_A = []
            Jaccard_B = []
            Jaccard_index_sum = []
            Jaccard_index_min = []

            kmeans_cluster_size = []
            kmeans_cluster_size_ordered = []
            for x in range(0, k):
                kmeans_cluster_size.append(len(kmeans_cluster[x]))
                kmeans_cluster_size_ordered.append(len(kmeans_cluster[x]))
            kmeans_cluster_size_ordered.sort(reverse=True)

            kmeans_cluster_order = []
            for y in range(0, k):
                kmeans_cluster_order.append(
                    kmeans_cluster_size.index(kmeans_cluster_size_ordered[y]))

            for z in range(0, k):
                b = kmeans_cluster_order[z]
                intersection_list = []
                intersection_length_list = []
                for c in range(0, k):
                    intersection = []
                    intersection = list(
                        set(kmeans_cluster[b]).intersection(MILP_cluster[c]))
                    intersection_list.append(intersection)
                    intersection_length_list.append(len(intersection))

                milpID = intersection_length_list.index(
                    max(intersection_length_list))

                while (milpID in kmeans_pairID):
                    intersection_length_list[milpID] = -1
                    milpID = intersection_length_list.index(
                        max(intersection_length_list))

                kmeans_pairID.append(milpID)
                kmeans_intersection.append(intersection_list[milpID])
                Jaccard_AB.append(intersection_length_list[milpID])
                Jaccard_A.append(len(kmeans_cluster[b]))
                Jaccard_B.append(len(MILP_cluster[milpID]))
                # jaccard index over sum
                Jaccard_index_sum.append(intersection_length_list[milpID]/(len(
                    kmeans_cluster[b])+len(MILP_cluster[milpID])-intersection_length_list[milpID]))
                if len(MILP_cluster[milpID]) != 0:
                    Jaccard_index_min.append(
                        intersection_length_list[milpID]/min(len(kmeans_cluster[b]), len(MILP_cluster[milpID])))
                else:
                    Jaccard_index_min.append(
                        intersection_length_list[milpID]/len(kmeans_cluster[b]))

            kmeans_pairID_list.append(kmeans_pairID)
            kmeans_intersection_list.append(kmeans_intersection)
            Jaccard_AB_list.append(Jaccard_AB)
            Jaccard_A_list.append(Jaccard_A)
            Jaccard_B_list.append(Jaccard_B)
            Jaccard_index_sum_list.append(Jaccard_index_sum)
            Jaccard_index_min_list.append(Jaccard_index_min)

            # visualize the overlap on a map
            matched_tracts = []
            for d in range(0, k):
                matched_tracts = matched_tracts + kmeans_intersection[d]

            matched_tract_df = Kmeans_result_df[k-2][f-2].copy()
            for index, row in matched_tract_df.iterrows():
                if (index in matched_tracts):
                    matched_tract_df.at[index, 'model'] = 1
                else:
                    matched_tract_df.at[index, 'model'] = 0

            print(str(k) + ' cluster, '+str(f)+' feature:')
            for e in range(0, k):
                print('Jaccard index (sum bottom) for cluster ' +
                      str(e+1)+' :'+str(Jaccard_index_sum[e]))
                print('Jaccard index (min bottom) for cluster ' +
                      str(e+1)+' :'+str(Jaccard_index_min[e]))
                print('Jaccard AnB for cluster ' +
                      str(e+1)+' :'+str(Jaccard_AB[e]))
                print('Jaccard A for cluster ' +
                      str(e+1)+' :'+str(Jaccard_A[e]))
                print('Jaccard B for cluster ' +
                      str(e+1)+' :'+str(Jaccard_B[e]))

            # yellow is not matched, green is matched tracts
            cluster_map(matched_tract_df, k,f,'matched_milp')

    return kmeans_pairID_list, kmeans_intersection_list, Jaccard_AB_list, Jaccard_A_list, Jaccard_B_list, Jaccard_index_sum_list, Jaccard_index_min_list

with open(f'{resultpath}bigmresults/results.pickle','wb') as f:
    pickle.dump(overlap(
    5, 5, MILP_result), f, protocol=4)

with open(f'{resultpath}bigmresults/milp.pickle','wb') as f:
    pickle.dump(MILP_result,f,protocol=4)
