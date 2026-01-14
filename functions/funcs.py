import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
import math
from scipy.signal import find_peaks
from fastdtw import fastdtw
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.preprocessing import minmax_scale
import matplotlib.transforms as tx
from matplotlib import pyplot as plt


def pose_detecting(video_path):

    # 3d points
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    print('pose detecting start...')
    for vid in video_path:
        if 'mp_pose' in vid:
            pass
        cap = cv2.VideoCapture(vid)
        dic = {'x': [], 'y': [], 'z': [], 'frame': [],
               'position': [], 'visibility': []}
        k_list = [str(m).split('.')[1].split(':')[0]
                  for m in list(mp_pose.PoseLandmark)]
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
                        x, y, z, c = [i.split(':') for i in str(
                            keypoints).split('\n')[:-1]]
                        dic[x[0]].append(float(x[1]))
                        dic[y[0]].append(float(y[1]))
                        dic[z[0]].append(-float(z[1]))
                        dic[c[0]].append(float(c[1]))
                        dic['position'].append(k_list[ix])
                        dic['frame'].append(f)
                except:
                    pass

        pd.DataFrame(dic).to_csv(
            f"./mp_posed/{vid.split('/')[-1][:-4]}_3d_pose.csv", index=False)
        print(f"{vid[:-4]}_3d_pose.csv generated")
        cap.release()
    print('pose detecting completed')


def outlier_iqr(data):
    # lower, upper 글로벌 변수 선언하기

    # 4분위수 기준 지정하기
    q25, q75 = np.quantile(data, 0.25), np.quantile(data, 0.75)

    # IQR 계산하기
    iqr = q75 - q25

    # outlier cutoff 계산하기
    cut_off = iqr * 1.5

    # lower와 upper bound 값 구하기
    lower, upper = q25 - cut_off, q75 + cut_off

    # print('IQR은', iqr, '이다.')
    # print('lower bound 값은', lower, '이다.')
    # print('upper bound 값은', upper, '이다.')
    return lower, upper


def transform_label(label):
    label = label.replace(' ', '_').replace('(', '').replace(')', '')
    if "left" in label:
        return "l" + label.replace("left", "")
    elif "right" in label:
        return "r" + label.replace("right", "")
    return label


def scale_coordinates(df, target_min=-0.9, target_max=0.9):
    # Calculate ranges for x and y
    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()
    z_range = df['z'].max() - df['z'].min()

    max_range = max(x_range, y_range, z_range)
    scale_factor = (target_max - target_min) / max_range

    # Apply scaling
    df['x'] = (df['x'] - df['x'].mean()) * scale_factor
    df['y'] = (df['y'] - df['y'].mean()) * scale_factor
    df['z'] = (df['z'] - df['z'].mean()) * scale_factor
    return df


def make_position_df(df):
    '''
        df = dataframe
        video_type = front or side
    '''
    dic = {}
    # df['position'] = df['position'].apply(transform_label)

    for pos in df.position.unique():
        dic[f'{pos}'] = df[df.position ==
                           pos].reset_index().drop(columns=['index'])
    return dic


def make_angle_df(df, pos1, pos2, pos3):

    m1 = df.position == pos1
    m2 = df.position == pos2
    m3 = df.position == pos3
    m = m1 | m2 | m3
    return df[m]


def split_array(arr, threshold, frames=30):
    """
    Split a NumPy array into sub-arrays based on a threshold condition.
    """
    diffs = np.abs(np.diff(arr))
    split_indices = np.flatnonzero(diffs > threshold) + 1
    return [np.array(arr[start:end])
            for start, end in zip([0] + list(split_indices), list(split_indices) + [len(arr)]) if end-start > frames]


def find_peaks_along_axis(arr, axis=-1):

    peaks = np.apply_along_axis(
        lambda x: find_peaks(-x, distance=17)[0], axis, arr)
    return peaks


# def get_cycles(peaks, lower):
#     diffs = np.diff(peaks)
#     cycles = diffs[diffs >= lower]
#     return cycles


def remove_close_peaks(peaks, diffs, quantile=0.1):
    threshold = np.quantile(diffs, quantile)
    peak_diffs = np.diff(peaks)
    mask = peak_diffs >= threshold
    result = list(
        set([peaks[0]] + list(np.array(peaks)[:-1][mask]) + [peaks[-1]]))
    return result

def calculate_3d_angle(a, b, c):
    a = np.array(a)  # 첫 번째 점
    b = np.array(b)  # 중심 점
    c = np.array(c)  # 세 번째 점
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)



def calculate_angle(joint_a, joint_b, joint_c):
    # Create arrays from the joints' coordinates
    a = np.array(joint_a)
    b = np.array(joint_b)
    c = np.array(joint_c)

    # Create vectors
    vec1 = abs(b - a)
    vec2 = abs(b - c)

    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Calculate the angle in radians and then convert to degrees
    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_deg = np.degrees(angle_rad)
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    return angle_deg


def fastdtw_two_data(tsd1, tsd2):
    # 여러 시계열 데이터
    time_series_data = [tsd1, tsd2]

    # 모든 시계열 쌍 간의 거리 계산
    for i in range(len(time_series_data)):
        for j in range(i+1, len(time_series_data)):
            distance, _ = fastdtw(time_series_data[i], time_series_data[j])
            # print(f"DTW Distance between series {i} and {j}: {distance}")
    return distance


def find_interp_index(special_number, x_new, ):
    if special_number >= x_new.min() and special_number <= x_new.max():
        # Find the index in the new x array closest to the special number
        special_index = (np.abs(x_new - special_number)).argmin()
        # Draw a vertical line at the special number's position in the interpolated space
        plt.axvline(x=x_new[special_index], color='red',
                    linestyle='--', label=f'Special Number at {special_number}')
        return special_index


def similarity_check(tsd1, tsd2, do_scale=False):
    if do_scale:
        tsd1 = minmax_scale(tsd1)
        tsd2 = minmax_scale(tsd2)

    cos_sim = cosine_similarity(tsd1.reshape(1, -1), tsd2.reshape(1, -1))
    dtw = fastdtw_two_data(tsd1, tsd2)
    corr, p = pearsonr(tsd1, tsd2)
    # correlation = tsd1.corr(tsd2)
    return cos_sim[0][0], dtw, corr, -np.log(p)


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
    stretched[np.where(stretched == 0)] = np.nan
    nans, x = nan_helper(stretched)
    stretched[nans] = np.interp(x(nans), x(~nans), stretched[~nans])
    return stretched


def get_front_section(is_front, sections, rhs, lhs, rto, lto):
    ixs = []
    for i, ix in enumerate(sections):
        if is_front:
            if i % 2 == 0:
                ixs.append(i)
        else:
            if i % 2 != 0:
                ixs.append(i)
    sections = [sections[ix] for ix in ixs]
    rhs = [rhs[ix] for ix in ixs]
    lhs = [lhs[ix] for ix in ixs]
    rto = [rto[ix] for ix in ixs]
    lto = [lto[ix] for ix in ixs]

    return sections, rhs, lhs, rto, lto


def plot_heel_data(sections, rhs, lhs,  dic):

    fig, axes = plt.subplots(len(sections),
                             1, figsize=(10, 10), sharex=True)
    for ax_ix, sec in enumerate(sections):
        ax = axes[ax_ix]
        right_heel_y = dic['R_HEEL'].y[sec]
        left_heel_y = dic['L_HEEL'].y[sec]
        ax.plot(minmax_scale(right_heel_y), label='right')
        ax.plot(minmax_scale(left_heel_y), label='left')

        trans = tx.blended_transform_factory(
            ax.transData, ax.transAxes)

        rhs_peaks = rhs[ax_ix]
        lhs_peaks = lhs[ax_ix]
        ax.plot(np.repeat(rhs_peaks, 3), np.tile([0, 1, np.nan], len(rhs_peaks)),
                linewidth=1, color='k', transform=trans, label='rhs')
        ax.plot(np.repeat(lhs_peaks, 3), np.tile([0, 1, np.nan], len(lhs_peaks)),
                linewidth=1, color='grey', transform=trans, label='lhs')
        ax.legend(loc='best')
        ax.set_title(f'sections {ax_ix}')
        fig.tight_layout()


def plot_skeletal_structure(df, save_path, frame, view='front'):
    """
    Plot the skeletal structure from pose estimation data.
    """
    mp_pose = mp.solutions.pose  # Initialize the MediaPipe Pose class
    pose_connections = mp_pose.POSE_CONNECTIONS

    # Filter for the initial frame (or another condition as needed)
    df_0 = df[df['frame'] == frame]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plotting front view scatter and connections
    axes[0].scatter(df_0['x'], df_0['y'])
    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(-1, 1)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title(f'{view.capitalize()} View')

    # Draw the connections between joints
    for connection in pose_connections:
        d = df_0.iloc[list(connection), :]
        axes[0].plot(d['x'], d['y'], c='black')

    # Plotting side view scatter and connections
    axes[1].scatter(df_0['z'], df_0['y'])
    axes[1].set_xlim(-1, 1)
    axes[1].set_ylim(-1, 1)
    axes[1].set_xlabel('Z')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Side View')

    for connection in pose_connections:
        d = df_0.iloc[list(connection), :]
        axes[1].plot(d['z'], d['y'], c='black')

    # Save the figure
    fig.savefig(os.path.join(save_path, 'skeleton.png'))
    plt.close(fig)
