from shared import np, pd, filepath, readindata_std

def pairwise_distance(n, LABEL, f, k, runtime=False):
    # n = number of tracts

    #read in data
    if runtime:
        distance_data = readindata_runtime(LABEL, f, k, n)
    else:
        distance_data = readindata_std(LABEL, f)
    distance_data = distance_data.iloc[:, 1:f+1]
    max_distance = 0
    for i in range(0, n):
        for j in range(0, n):
            t_1 = distance_data.iloc[i, :].values  # tract i features
            t_2 = distance_data.iloc[j, :].values  # tract j features
            dist = np.linalg.norm(t_1-t_2)
            if dist > max_distance:
                max_distance = dist
    return max_distance

def readindata_runtime(LABEL,f,k,numTracts):
    # create data for new york for piecewise regression model
    piecewise_data = pd.read_csv(filepath+"synthetic_MIP_runtime.csv")    
    if f==2:
        kp_feature_list = ['high school or less','white, not hispanic']
    elif f==3: 
        kp_feature_list = ['high school or less','white, not hispanic','college degree +']
    elif f==4:
        kp_feature_list = ['high school or less','white, not hispanic','college degree +','professional employee']
    elif f==5:
        kp_feature_list = ['high school or less','white, not hispanic','college degree +','professional employee','female-headed fam with children']
    feature_list = ['tractid'] + kp_feature_list
    piecewise_data = piecewise_data[feature_list]
    
    # select first numTracts rows of data
    piecewise_data = piecewise_data.iloc[0:numTracts]
    for i in range(0,k):
        start = int(i*(numTracts/k))
        end = int((i+1)*(numTracts/k))
        if i==0:
            label_list = piecewise_data.iloc[start:end,(i+1)].values.flatten().tolist()
        else:
            label_list = label_list+ piecewise_data.iloc[start:end,(i+1)].values.flatten().tolist()

    piecewise_data=piecewise_data.assign(change_incpc=label_list)
    return piecewise_data