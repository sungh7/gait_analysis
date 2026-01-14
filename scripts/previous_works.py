import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm

import mediapipe as mp

from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import minmax_scale

import matplotlib.pyplot as plt
import matplotlib.transforms as tx
import seaborn as sb

from functions import etc_funcs

def pose_detecting(video_path):
    #3d points
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    print('pose detecting start...')
    for vid in video_path:
        if 'mp_pose' in vid: pass
        cap = cv2.VideoCapture(vid)
        dic = {'x':[], 'y':[], 'z':[], 'frame':[], 'position':[], 'visibility':[]}
        k_list = [str(m).split('.')[1].split(':')[0] for m in list(mp_pose.PoseLandmark)]
        print(f'{vid} is detecting...')
        
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            for f in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                try:
                    _, image = cap.read()
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)

                    for ix, keypoints in enumerate(results.pose_world_landmarks.landmark):
                        x, y, z, c =  [i.split(':') for i in str(keypoints).split('\n')[:-1]]
                        dic[x[0]].append(float(x[1]))
                        dic[y[0]].append(-float(y[1]))
                        dic[z[0]].append(-float(z[1]))
                        dic[c[0]].append(float(c[1]))
                        dic['position'].append(k_list[ix])
                        dic['frame'].append(f)
                except:
                    pass

        pd.DataFrame(dic).to_csv(f"./mp_posed/{vid.split('/')[-1][:-4]}_3d_pose.csv", index=False)
        print(f"./mp_posed/{vid[:-4]}_3d_pose.csv generated")
        cap.release()
    print('pose detecting completed')


mp_pose = mp.solutions.pose
for vid in li:
    # if 'mp_pose' in vid: pass
    cap = cv2.VideoCapture(vid)
    dic = {'x':[], 'y':[], 'z':[], 'frame':[], 'position':[], 'c':[]}
    k_list = [str(m).split('.')[1].split(':')[0] for m in list(mp_pose.PoseLandmark)]
    print(f'{vid} is detecting...')
    
    with mp_pose.Pose(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as pose:

        for f in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            try:
                _, image = cap.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                is_head_above = False
                for ix, keypoint in enumerate(results.pose_world_landmarks.landmark):
                    x, y, z, c = keypoint.x, keypoint.y, keypoint.z, keypoint.visibility
                    
                    if ix == 0 and y < 0:
                        is_head_above = True
                        
                    y = -y if is_head_above else y 
                    dic['x'].append(x)
                    dic['y'].append(y)
                    dic['z'].append(z)
                    dic['c'].append(c)
                    dic['position'].append(k_list[ix])
                    dic['frame'].append(f)
            except:
                pass

df = pd.DataFrame(dic)
df_front = pd.read_csv('전공의gait/1-1_front_3d_pose.csv')
df_side = pd.read_csv('전공의gait/1-2_side_3d_pose.csv')
df_front['x'] = df_front['x'].apply(lambda x: -(x))

def make_position_df(df, video_type=''):
    '''
        df = dataframe
        video_type = front or side
    '''
    for pos in df.position.unique():
        # if pos.split('_', 1)[-1] in ['LEFT', 'RIGHT']:
        #     pos = '_'.join(pos.split('_'))
        #     globals()[f'df_{pos}'] = df_front[df_front.position==pos]
        globals()[f'df_{video_type}_{pos}'] = df[df.position==pos].reset_index()


def make_angle_df(df, pos1, pos2, pos3):
    
    m1 = df.position==pos1
    m2 = df.position==pos2
    m3 = df.position==pos3
    m = m1 | m2 | m3
    return df[m]


mp_pose = mp.solutions.pose
df_0 = df_front[df_front.frame==0]
fig, axes = plt.subplots(1, 2)
axes[0].scatter(df_0.x, df_0.y)
axes[0].set_xlim(-1,1)
axes[0].set_ylim(-1,1)
axes[0].set_title('frontal')
for i in mp_pose.POSE_CONNECTIONS:
    d = df_0.iloc[list(i), :]
    axes[0].plot(d.x, d.y, c='black')


axes[1].scatter(df_0.z, df_0.y)
axes[1].set_xlim(-1,1)
axes[1].set_ylim(-1,1)
axes[1].set_title('sagittal')
for i in mp_pose.POSE_CONNECTIONS:
    d = df_0.iloc[list(i), :]
    axes[1].plot(d.z, d.y, c='black')

for i in df_front.position.unique():
    if i.split('_'):
        ''.join(i.split('_')[1:])
posit = list(set([
    '_'.join(i.split('_', 1)) 
    for i in df_front.position.unique()
    if i.split('_')]))        

pos_set = set([pos.split('_', 2)[-1] for pos in posit]) - {'INNER', 'OUTER'}

for pos in pos_set:
    if pos not in  ["NOSE", "LEFT", "RIGHT"]:
        globals()[f'{pos}_mean'] = pd.DataFrame(zip(eval(f'df_front_LEFT_{pos}').y.values, eval(f'df_front_RIGHT_{pos}').y.values)).apply(lambda x: np.mean(x), axis=1)
    else: pass

df_filtered = df_front.frame.unique()[wrong_detect_masking[~wrong_detect_masking].index.values]    
df_front_NOSE = df_front_NOSE.iloc[df_filtered]

nose_m1 = df_front_NOSE.x > lower
nose_m2 = df_front_NOSE.x < upper
nose_m3 = nose_m1&nose_m2
df_front_NOSE[nose_m3]

xlength_hip = []
for frame in df_front.frame.unique():
    df = pd.concat([df_front_LEFT_HIP, df_front_RIGHT_HIP])
    xlength_hip.append(abs(df.iloc[0,0] - df.iloc[1,0]))

xlength_sh = []
for frame in df_front.frame.unique():
    df = pd.concat([df_front_LEFT_SHOULDER, df_front_RIGHT_SHOULDER])
    xlength_sh.append(abs(df.iloc[0,0] - df.iloc[1,0]))

xlength_ear = []
for frame in df_front.frame.unique():
    df = pd.concat([df_front_LEFT_EAR, df_front_RIGHT_EAR])
    xlength_ear.append(abs(df.iloc[0,0] - df.iloc[1,0]))
    
width_ear_diff = np.array([abs(xlength_ear[i]-xlength_ear[i+1]) for i in range(len(xlength_ear)-1)])
width_hip_diff = np.array([abs(xlength_hip[i]-xlength_hip[i+1]) for i in range(len(xlength_hip)-1)])
width_sh_diff  = np.array([abs(xlength_sh[i]-xlength_sh[i+1]) for i in range(len(xlength_sh)-1)])

diff_ = pd.DataFrame([width_ear_diff+width_hip_diff+width_sh_diff]).T

width_m1 = diff_[0] > lower
width_m2 = diff_[0] < upper
width_m3 = width_m1&width_m2


a = 0
tdata = diff_[width_m3].index.values
tdata_section = []
for i in range(tdata.size-1):
    if abs(tdata[i]- tdata[i+1]) > 5:
        tdata_section.append(tdata[a:i+1])
        a = i+1
    elif i+1==tdata.size-1:
        tdata_section.append(tdata[a:])
    
inter_data = np.array(list(set(normal_data) & set(tdata)))

intersection = []
a = 0
for i in range(len(inter_data)-1):
    if abs(inter_data[i] - inter_data[i+1]) > 10:
        intersection.append(inter_data[a:i+1])
        a = i+1
        print(a)
    elif i+1 > len(inter_data):
        intersection.append(inter_data[a:])
intersection = [i for i in intersection if i.size > 10]
if len(intersection) <1: intersection = [inter_data]
# %matplotlib widget
lhs = []
rhs = [] 
fig, axes = plt.subplots(3, len(intersection))
for i, sec in enumerate(intersection):
    for ix, d in enumerate([df_front_RIGHT_HEEL.iloc[sec],df_front_LEFT_HEEL.iloc[sec]]):
        for iax, axis in enumerate(['x','y','z']):
            ax = axes[iax, i] if len(axes.shape)==2 else axes[iax]
            data = d.reset_index()[axis]
            # data = savgol_filter(data, 17, 3)
            ax.plot(data)
            peaks, _ = find_peaks(-data)
            ax.plot(peaks, data[peaks], "x")
            if axis=='y':
                if ix == 0:
                    rhs.append(peaks)
                elif ix == 1:
                    lhs.append(peaks)  
new_rhs = []
for hs in rhs:
    will_remove_index = []
    for ix, item in enumerate(hs[:-1]):
        if abs(item - hs[ix+1]) < np.quantile(rhs_diffs, 0.2):
            will_remove_index.append(ix+1)
    new_rhs.append(np.delete(hs, will_remove_index))
%matplotlib inline
fig, axes = plt.subplots(1, 1, figsize=(15,5))
for ix, sec in enumerate(intersection):
    ax = axes
    ax.plot(minmax_scale(df_front_RIGHT_HEEL.y[sec]), label='front')
    # ax.plot(minmax_scale(df_side_RIGHT_HEEL.y[sec]), label='side')
    ax.plot(minmax_scale(180-np.array(leg_r_angles[sec])), label='side_angle')
    trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(np.repeat(rhs[ix], 3), np.tile([0, 1, np.nan], len(rhs[ix])), linewidth=2, color='k', transform=trans)
    ax.legend(loc='best')
fig.tight_layout()            
rhs_gc = []
for ix, sec in enumerate(intersection):
    t_list = []
    for i in range(len(new_rhs[ix])-1):
        start, end = new_rhs[ix][i], new_rhs[ix][i+1]
        t_list.append(sec[start:end])
    rhs_gc.append(t_list)
for ix, gc in enumerate(rhs_gc):
    if ix % 2 == 0:
        fig, ax = plt.subplots(1,1,)
        if len(gc) > 3:
            for j in range(len(gc)):
                ax.plot(180-np.array(leg_r_angles[gc[j]]))

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def stretched_data(y_data, stretch_size, reverse=False):
    y = np.array(y_data)
    if reverse:
        y = y[::-1]
    every = stretch_size / y.size
    stretched = np.zeros(100)

    for hd in range(y.size):
        stretched[round(hd*every)] = y[hd]

    stretched[-1] = y[-1]
    stretched[np.where(stretched==0)] = np.nan
    nans, x = nan_helper(stretched)
    stretched[nans] = np.interp(x(nans), x(~nans), stretched[~nans])
    return stretched

leg_r_angles = 180 - leg_r_angles


ang_data_of_gc = []
fig, ax = plt.subplots(1,1,)
for ix, gc in enumerate(rhs_gc):
    if ix % 2 == 0:
        for j in range(len(gc)):
            y = stretched_data(leg_r_angles[gc[j]], 100)
            ax.plot(y, )
            ang_data_of_gc.append(y)
            
t = np.zeros(100)
for i in ang_data_of_gc:
    t += i

ax.plot(t/len(ang_data_of_gc), color='k')

t = np.zeros(100)
for i in ang_data_of_gc:
    t += i
ax = plt.gca()
ax.plot(t/len(ang_data_of_gc))
%matplotlib inline
fig, axes = plt.subplots(6, 1, figsize=(15,15))
for ix, sec in enumerate(intersection):
    ax = axes[ix]
    # ax.plot((180-np.array(leg_l_angles)[sec]), label='front')
    ax.plot(minmax_scale(df_front_LEFT_HEEL.y[sec]), label='side')
    trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(np.repeat(new_lhs[ix], 3), np.tile([0, 1, np.nan], len(new_lhs[ix])), linewidth=2, color='k', transform=trans)
    ax.legend(loc='best')
fig.tight_layout()