def outlier_iqr(data, column): 
    import numpy as np
    # lower, upper 글로벌 변수 선언하기     
    global lower, upper    
    
    # 4분위수 기준 지정하기     
    q25, q75 = np.quantile(data[column], 0.25), np.quantile(data[column], 0.75)          
    
    # IQR 계산하기     
    iqr = q75 - q25    
    
    # outlier cutoff 계산하기     
    cut_off = iqr * 1.5          
    
    # lower와 upper bound 값 구하기     
    lower, upper = q25 - cut_off, q75 + cut_off     
    
    print('IQR은',iqr, '이다.')     
    print('lower bound 값은', lower, '이다.')     
    print('upper bound 값은', upper, '이다.')    
    
    # 1사 분위와 4사 분위에 속해있는 데이터 각각 저장하기     
    data1 = data[data[column] > upper]     
    data2 = data[data[column] < lower]    
    
    # 이상치 총 개수 구하기
    return print('총 이상치 개수는', data1.shape[0] + data2.shape[0], '이다.')



def cal_angle(data, x, y):
    import numpy as np
    
    df = data.loc[:, [x,y]]
    df = df.apply(lambda x: round(x*1000))
    a = df.iloc[0].values
    b = df.iloc[1].values
    c = df.iloc[2].values
    
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



def f_peaks(y, ax):
    from scipy.signal import find_peaks
    '''def f_peaks(y, ax):
    peaks, _ = find_peaks(y)
    ax.plot(peaks, y[peaks], "x")'''
    peaks, _ = find_peaks(y)
    ax.plot(peaks, y[peaks], "x")