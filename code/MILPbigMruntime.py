#!/usr/bin/env python
# coding: utf-8
from gurobipy import *
from shared import *
from milpshared import *

def MIP_model_BigM(LABEL, numTracts, numModels, numFeatures, runtimelimit, M_val):
    # read in feature value and label value from dataframe
    DF = readindata_runtime(LABEL, numFeatures, numModels, numTracts)
    df = DF.copy()

    M = M_val

    # feature data
    # create feature value list Xij
    X_val = df.iloc[:, 1:numFeatures+1].values.tolist()
    Y = df.iloc[:, -1].tolist()  # create label value list Yi

    model = Model()
    runtime = 0

    # Add variables
    X = {}
    E = {}
    W = {}
    B = {}
    C = {}
    Z = {}
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

    for j in range(numFeatures):
        for k in range(numModels):
            Z[(j, k)] = model.addVar(vtype=GRB.CONTINUOUS, name="Z%d,%d" % (j, k))

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

    for j in range(numFeatures):
        for k in range(numModels):
            model.addConstr(W[(j, k)] <= Z[(j, k)])

    for j in range(numFeatures):
        for k in range(numModels):
            model.addConstr(-W[(j, k)] <= Z[(j, k)])

 # set objective
    model.setObjective( quicksum( quicksum( E[(i,k)] for i in range(numTracts)) for k in range(numModels)))

    model.Params.timeLimit = runtimelimit  # 12 hours
#     model.Params.LogFile = filepath+"MIP_bigM_real_log_m"+str(numModels)+"_f"+str(numFeatures)
    model.optimize()
#     model.write(filepath+"MIP_bigM_real_m"+str(numModels)+"_f"+str(numFeatures)+".sol")

    df = pd.DataFrame(columns=['Dec_Var', 'Val'])
    for v in model.getVars():
    #     if runtimelimit >20: #1800
    #         print(v.varName, v.x)

        df = df.append({'Dec_Var': v.varName, 'Val': v.x}, ignore_index=True)
    # print('Obj:', model.objVal)
    # if runtimelimit >20: #1800
    #     print("Final MIP gap value in percentage: %f " % (model.MIPGap*100))

    error_list = []
    error_list = [x.X for x in model.getVars() if x.VarName.find('E') != -1]

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

    return df, MAE, MSE, bias_list, coef_list, model.MIPGap*100, model.runtime


def collect_result(K, F):
    # k rows, f columns (k = # of clusters, f = # of features
    MSElist = []
    MAElist = []
    resultlist = []

    for k in tqdm(range(2, K+1)):
        MSElist_sameCluster = []
        MAElist_sameCluster = []
        resultlist_sameCluster = []

        for f in range(2, F+1):
            numtract_list = []

            # setting the lower bound and upper bound on number of tracts (can change this)
            for i in myrange(240, 1200, 240):
                try:
                # run the MILP model
                    M_val = pairwise_distance(i, 'incpc', f, k,runtime=True)
                    result, MAE, MSE, _, _, _, _ = MIP_model_BigM(
                        'incpc', i, k, f, 7200, M_val)
                except Exception as e:
                    print(e)
                    result = []
                    MAE = []
                    MSE = []
                    i = -1
                    pass
                resultlist_sameCluster.append(result)
                MAElist_sameCluster.append(MAE)
                MSElist_sameCluster.append(MSE)
                numtract_list.append(i)
                # runtime_list.append(runtime)

                # plt.figure(figsize=(10,10))
                # plt.title('Runtime vs Number of Tracts for MILP, '+str(k)+' Clusters &'+ str(f)+' Features')
                # plt.xlabel('Number of Tracts')
                # plt.ylabel('Runtime')
                # plt.plot(numtract_list, runtime_list)
                # plt.savefig(f'{logpath}bigmruntime{i}.pdf',dpi=300)
        MAElist.append(MAElist_sameCluster)
        MSElist.append(MSElist_sameCluster)
        resultlist.append(resultlist_sameCluster)
    return MSElist, MAElist, (None, None), resultlist

# I set an upper limit for runtime of 7200 seconds (2 hrs), if runtime = 7200, it means MILP did not run to optimality
# in 2 hours


bigmresults = display_result(5, 5, 'bigm_runtime', collect_result)
bigmresults = fixkmeansresultslist(bigmresults)
out = get_silhouette_values(bigmresults)
print(sorted(out, key=lambda x: x[2]))

with open(logpath + 'bigm_runtime.dill', 'wb') as f:
    dill.dump_session(f)
