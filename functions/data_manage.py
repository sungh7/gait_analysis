import numpy as np
import pandas as pd

from functions.etc_funcs import cal_angle
def position_df_to_dic(df, video_type=''):
    
    '''
        df = dataframe
        video_type = front or side
    '''
    df_dic = {}
    for pos in df.position.unique():
        df_dic[f'df_{video_type}_{pos}'] = df[df.position==pos].reset_index()
    
    return df_dic
def angle_df_to_dic(df, pos1, pos2, pos3, direction=''):
    if len(direction) > 0:
         pos1, pos2, pos3 = [f'{direction}_{i}' for i in [pos1, pos2, pos3]]
    m1 = df.position==pos1
    m2 = df.position==pos2
    m3 = df.position==pos3
    m = m1 | m2 | m3
    return df[m]

def outlier_iqr(data, column): 
    
    # 4분위수 기준 지정하기     
    q25, q75 = np.quantile(data[column], 0.25), np.quantile(data[column], 0.75)          
    
    # IQR 계산하기     
    iqr = q75 - q25    
    
    # outlier cutoff 계산하기     
    cut_off = iqr * 1.5          
    
    # lower와 upper bound 값 구하기     
    lower, upper = q25 - cut_off, q75 + cut_off     
    
    # 1사 분위와 4사 분위에 속해있는 데이터 각각 저장하기     
    data1 = data[data[column] > upper]     
    data2 = data[data[column] < lower]    
    
    # 이상치 총 개수 구하기
    return lower, upper

def calculate_xlength_diff(df_left, df_right, col='x'):
        
    diff = abs(df_left[col] - df_right[col]) 
    return np.diff(diff)  

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def stretched_data(y_data, stretch_size, reverse=False):
    y = np.array(y_data)
    if reverse:
        y = y[::-1]
    every = stretch_size / y.size
    stretched = np.zeros(stretch_size)

    for hd in range(y.size):
        stretched[round(hd*every)] = y[hd]

    stretched[-1] = y[-1]
    stretched[np.where(stretched==0)] = np.nan
    nans, x = nan_helper(stretched)
    stretched[nans] = np.interp(x(nans), x(~nans), stretched[~nans])
    return stretched


# Assuming the make_angle_df, cal_angle, and stretched_data functions are defined elsewhere
# Assuming df_front, lhs_gc, and rhs_gc are defined as per your context

def get_joint_angle(df, x='z', y='y'):
    
    # Calculate joint angles for each unique frame in the DataFrame
    return [cal_angle(df[df.frame == frame], x, y) for frame in df.frame.unique()]

