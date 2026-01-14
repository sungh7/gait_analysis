import json, glob, sys, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
# from scipy.stats import describe


def decomposit_val(val, trigger):
    x, y = val
    if trigger == 0:
        return x
    elif trigger == 1:
        return y
    else:
        pass
    
def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

def cal_angle(a, b, c):
    
    a = np.array([a.iloc[:, 0], a.iloc[:, 1]])
    b = np.array([b.iloc[:, 0], b.iloc[:, 1]])
    c = np.array([c.iloc[:, 0], c.iloc[:, 1]])
    
    aPoint = a-b
    bPoint = c-b
    
    angA = np.arctan2(*aPoint[::-1])
    angB = np.arctan2(*bPoint[::-1])
    ang = np.rad2deg((angA-angB)%(2*np.pi)) 
    
    if ang > 180 :
        angResult = 360 - ang
    else:
        angResult = ang
    return angResult

def cal_angle2(a, b, c):
    a = np.array([a.x1, a.y1])
    b = np.array([b.x1, b.y1])
    c = np.array([c.x1, c.y1])
    
    aPoint = a-b
    bPoint = c-b
    
    angA = np.arctan2(*aPoint[::-1])
    angB = np.arctan2(*bPoint[::-1])
    ang = np.rad2deg((angA-angB)%(2*np.pi)) 
    return ang

def makeAngDf(dfFiltered):
    ang = {
         'RshouldUp': [],
         'LshouldUp': [],
         'RshouldBotIn': [],
         'LshouldBotIn': [],
         'RpelvisUp': [],
         'LpelvisUp': [],
         'RpelvisBot': [],
         'LpelvisBot': [],
         'RshouldBotOutS': [],
         'LshouldBotOutS': [],
         'RshouldBotOutL': [],
         'LshouldBotOutL': [],
         'Rarm': [],
         'Larm': [],
         'RpevisUp': [],
         'LpevisUp': [],
         'Rleg': [],
         'Lleg': []
    }
    for i in dfFiltered.frame.unique():
        df = dfFiltered[dfFiltered['frame']==i]
        ang['RshouldUp'].append(cal_angle(df.iloc[0], df.iloc[1], df.iloc[2]))
        ang['LshouldUp'].append(cal_angle(df.iloc[0], df.iloc[1], df.iloc[3]))
        ang['RshouldBotIn'].append(cal_angle(df.iloc[4], df.iloc[1], df.iloc[2]))
        ang['LshouldBotIn'].append(cal_angle(df.iloc[4], df.iloc[1], df.iloc[3]))
        ang['RpelvisUp'].append(cal_angle(df.iloc[1], df.iloc[4], df.iloc[5]))
        ang['LpelvisUp'].append(cal_angle(df.iloc[1], df.iloc[4], df.iloc[6]))
        ang['RpelvisBot'].append(cal_angle(df.iloc[4], df.iloc[5], df.iloc[7]))
        ang['LpelvisBot'].append(cal_angle(df.iloc[4], df.iloc[6], df.iloc[8]))
        ang['RshouldBotOutS'].append(cal_angle(df.iloc[1], df.iloc[3], df.iloc[11]))
        ang['LshouldBotOutS'].append(cal_angle(df.iloc[1], df.iloc[2], df.iloc[9]))
        ang['RshouldBotOutL'].append(cal_angle(df.iloc[1], df.iloc[3], df.iloc[12]))
        ang['LshouldBotOutL'].append(cal_angle(df.iloc[1], df.iloc[2], df.iloc[10]))
        ang['Rarm'].append(cal_angle(df.iloc[2], df.iloc[9], df.iloc[10]))
        ang['Larm'].append(cal_angle(df.iloc[3], df.iloc[11], df.iloc[12]))
        ang['RpevisUp'].append(cal_angle(df.iloc[4], df.iloc[5], df.iloc[13]))
        ang['LpevisUp'].append(cal_angle(df.iloc[4], df.iloc[6], df.iloc[14]))
        ang['Rleg'].append(cal_angle(df.iloc[5], df.iloc[7], df.iloc[13]))
        ang['Lleg'].append(cal_angle(df.iloc[6], df.iloc[8], df.iloc[14]))

    df_ang = pd.DataFrame(data=ang)

    #df_ang = mean_norm(df_ang)

    y1 = []
    for ix, i in enumerate(df_ang.columns):
        for j in df_ang[i]:
            y1.append(j)

    dfAng = pd.DataFrame({'y':y1})

    (a, b) = df_ang.shape
    position = [x for x in df_ang.columns for j in range(a) ]
    dfAng['position'] = position

    dfAng['frame'] = [j+1 for x in df_ang.columns for j in range(a)]
    return dfAng

def makeAngDf2(dfFiltered):
    ang = {
         'Nose_Neck_Rshoulder': [],
         'Nose_Neck_Lshoulder': [],
         'Rshoulder_Neck_Midhip': [],
         'Lshoulder_Neck_Midhip': [],
         'Neck_Midhip_Rhip': [],
         'Neck_Midhip_Lhip': [],
         'Midhip_Rhip_Rknee': [],
         'Midhip_Lhip_Lknee': [],
         'Neck_Lshoulder_Lelbow': [],
         'Neck_Rshoulder_Relbow': [],
         'Neck_Lshoulder_Lwrist': [],
         'Neck_Rshoulder_Rwrist': [],
         'Rshoulder_Relbow_Rwrist': [],
         'Lshoulder_Lelbow_Lwrist': [],
         'Midhip_Rhip_Rankle': [],
         'Midhip_Lhip_Lankle': [],
         'Rhip_Rknee_Rankle': [],
         'Lhip_Lknee_Lankle': []
    }
    for i in dfFiltered.frame.unique():
        df = dfFiltered[dfFiltered['frame']==i]
        ang['Nose_Neck_Rshoulder'].append(cal_angle2(df.iloc[0], df.iloc[1], df.iloc[2]))
        ang['Nose_Neck_Lshoulder'].append(cal_angle2(df.iloc[0], df.iloc[1], df.iloc[3]))
        ang['Rshoulder_Neck_Midhip'].append(cal_angle2(df.iloc[4], df.iloc[1], df.iloc[2]))
        ang['Lshoulder_Neck_Midhip'].append(cal_angle2(df.iloc[4], df.iloc[1], df.iloc[3]))
        ang['Neck_Midhip_Rhip'].append(cal_angle2(df.iloc[1], df.iloc[4], df.iloc[5]))
        ang['Neck_Midhip_Lhip'].append(cal_angle2(df.iloc[1], df.iloc[4], df.iloc[6]))
        ang['Midhip_Rhip_Rknee'].append(cal_angle2(df.iloc[4], df.iloc[5], df.iloc[7]))
        ang['Midhip_Lhip_Lknee'].append(cal_angle2(df.iloc[4], df.iloc[6], df.iloc[8]))
        ang['Neck_Lshoulder_Lelbow'].append(cal_angle2(df.iloc[1], df.iloc[3], df.iloc[11]))
        ang['Neck_Rshoulder_Relbow'].append(cal_angle2(df.iloc[1], df.iloc[2], df.iloc[9]))
        ang['Neck_Lshoulder_Lwrist'].append(cal_angle2(df.iloc[1], df.iloc[3], df.iloc[12]))
        ang['Neck_Rshoulder_Rwrist'].append(cal_angle2(df.iloc[1], df.iloc[2], df.iloc[10]))
        ang['Rshoulder_Relbow_Rwrist'].append(cal_angle2(df.iloc[2], df.iloc[9], df.iloc[10]))
        ang['Lshoulder_Lelbow_Lwrist'].append(cal_angle2(df.iloc[3], df.iloc[11], df.iloc[12]))
        ang['Midhip_Rhip_Rankle'].append(cal_angle2(df.iloc[4], df.iloc[5], df.iloc[13]))
        ang['Midhip_Lhip_Lankle'].append(cal_angle2(df.iloc[4], df.iloc[6], df.iloc[14]))
        ang['Rhip_Rknee_Rankle'].append(cal_angle2(df.iloc[5], df.iloc[7], df.iloc[13]))
        ang['Lhip_Lknee_Lankle'].append(cal_angle2(df.iloc[6], df.iloc[8], df.iloc[14]))

    df_ang = pd.DataFrame(data=ang)

    df_ang = mean_norm(df_ang)

    y1 = []
    for ix, i in enumerate(df_ang.columns):
        for j in df_ang[i]:
            y1.append(j)

    dfAng = pd.DataFrame({'angle':y1})

    (a, b) = df_ang.shape
    position = [x for x in df_ang.columns for j in range(a) ]
    dfAng['position'] = position

    dfAng['frame'] = [j+1 for x in df_ang.columns for j in range(a)]
    return dfAng

def jsonPreprocessing(path):
    path_ = glob.glob(path+'/*')
    path_ = sorted(path_)
    json_data= {}
    
    for i in range(25):
        json_data[str(i)] = []
        
    for l in path_:
        with open(l, 'r') as f:
            j = json.load(f)
            for i in range(25):
                for k in range(len(j['people'])):
                    json_data[str(i)].append(tuple(j['people'][k-1]['pose_keypoints_2d'][3*i:3*i+2]))

    df = pd.DataFrame(json_data)
    df_proc = df.loc[:,['0','1','2','5','8','9','12','10','13','3','4','6','7','11','14','17','18']]
    df_proc.columns=['Nose', 'Neck', 'Rshoulder', 'Lshoulder', 'MidHip', 'RHip','LHip','Rknee','Lknee','RElbow','RWrist','LElbow','LWrist','RAnkle','LAnkle','REar','LEar']
    df_proc = df_proc[~df_proc.isin([(0,0)])].dropna()
    
    x1 = []
    y1 = []
    for ix, i in enumerate(df_proc.columns):
        for j in df_proc[i]:
            try:
                x, y = j
                x1.append(-x)
                y1.append(-y)
            except:
                pass

    df_for_sb = pd.DataFrame({'x1':x1, 'y1':y1})
    a, b = df_proc.shape
    x, y = df_for_sb.shape
    position = [x for x in df_proc.columns for j in range(a) ]
    df_for_sb['position'] = position
    df_for_sb['frame'] = [j+1 for x in df_proc.columns for j in range(a)]

    dict_x = {}
    for col in df_proc.columns:
        dict_x[col] = df_proc.apply(lambda x : decomposit_val(x[col],0), axis=1)
    tdf_x = pd.DataFrame(dict_x)

    dict_y = {}
    for col in df_proc.columns:
        dict_y[col] = df_proc.apply(lambda x : decomposit_val(x[col], 1), axis=1)
    tdf_y = pd.DataFrame(dict_y)

    filter_ = tdf_x.Rshoulder[tdf_x.Rshoulder < tdf_x.Lshoulder].index
    filter_ = [i for i in tdf_x.index if i in list(filter_)]
    tdf_x_filtered = tdf_x.loc[np.array(filter_)]
    tdf_y_filtered = tdf_y.loc[np.array(filter_)]
    df_proc_filtered = df_proc.loc[np.array(filter_)]

    x1 = []
    y1 = []
    for ix, i in enumerate(df_proc_filtered.columns):
        for j in df_proc_filtered[i]:
            try:
                x, y = j
                x1.append(-x)
                y1.append(-y)
            except:
                pass
    df_test2 = pd.DataFrame({'x1':x1, 'y1':y1})
    (a, b) = df_proc_filtered.shape
    position = [x for x in df_proc_filtered.columns for j in range(a) ]
    x,y = df_test2.shape
    df_test2['position'] = position
    df_test2['frame'] = [j+1 for x in df_proc_filtered.columns for j in range(a)]

    
    Nose_list = []
    videoLabel = []

    num = path.split('/')[-1][5:]

    for j in tdf_x_filtered.Nose.values : Nose_list.append(j), videoLabel.append(num)

    noseSeries = pd.Series(Nose_list)
    videoLabel = pd.Series(videoLabel)
 
    bins = list(range(0, 1900, 100))
    label = [str(x)+'-'+str(x+100) for x in bins[:-1]]

    noseCut = pd.cut(noseSeries, bins, right=False, labels=label)

    df = pd.DataFrame([noseSeries, noseCut, videoLabel]).T

    df.columns = ['data','label','num']

    count = []
    for i in df.num.unique():
        tdf = df[df.num == i]
        for j in tdf.label.value_counts()[:5].index:
            count.append(j)

    standard = list(pd.Series(count).value_counts().index[0])

    standard.remove('-')

    slicing = int(len(standard)/2)

    dfNose = df_test2[df_test2.position == 'Nose']

    m1 = dfNose.x1 <= -int(''.join(standard[:slicing]))
    m2 = dfNose.x1 >= -int(''.join(standard[slicing:]))
    m  = m1 & m2

    noseFilter = list(dfNose[m].frame)

    dfFiltered = df_test2[df_test2.frame.isin(noseFilter)]
    
    return dfFiltered