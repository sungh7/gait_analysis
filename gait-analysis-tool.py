import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

class GaitAnalysisTool:
    def __init__(self):
        self.fps = 30  # 영상의 FPS, 필요에 따라 조정
        self.pose_data = []
        self.current_segment = []

    def add_frame(self, frame_data):
        self.pose_data.append(frame_data)
        self.current_segment.append(frame_data)
        
        # 일정 길이 이상의 세그먼트가 쌓이면 분석 수행
        if len(self.current_segment) >= self.fps * 2:  # 최소 2초
            return self.analyze_current_segment()
        return None

    def analyze_current_segment(self):
        segment_data = np.array(self.current_segment)
        
        # z축 방향의 움직임 계산 (앞뒤 움직임)
        hip_center = (segment_data[:, 23] + segment_data[:, 24]) / 2  # 양쪽 힙의 중간점
        movement = np.diff(hip_center[:, 2])
        
        # 노이즈 제거를 위한 스무딩
        smoothed_movement = gaussian_filter1d(movement, sigma=2)

        # 방향 변화 감지
        direction_changes = np.diff(np.sign(smoothed_movement))
        if np.any(direction_changes != 0):
            # 방향 변화가 감지되면 세그먼트 초기화
            self.current_segment = []
            return None

        # 걸음 주기 감지
        left_ankle_y = segment_data[:, 27, 1]  # 27은 'left ankle'의 인덱스
        peaks, _ = find_peaks(left_ankle_y, distance=15)  # 최소 0.5초 간격

        # 분석 수행
        cadence = self.calculate_cadence(peaks, len(segment_data) / self.fps)
        stride_length = self.calculate_stride_length(peaks, segment_data)
        walking_speed = self.calculate_walking_speed(cadence, stride_length)
        hip_angle, knee_angle, ankle_angle = self.calculate_joint_angles(segment_data)

        return {
            "cadence": cadence,
            "stride_length": stride_length,
            "walking_speed": walking_speed,
            "hip_angle": np.mean(hip_angle),
            "knee_angle": np.mean(knee_angle),
            "ankle_angle": np.mean(ankle_angle)
        }

    def calculate_cadence(self, gait_cycles, segment_duration):
        steps = len(gait_cycles)
        return (steps / segment_duration) * 60  # 분당 걸음 수

    def calculate_stride_length(self, gait_cycles, segment_data):
        stride_lengths = []
        for i in range(len(gait_cycles) - 2):
            start = gait_cycles[i]
            end = gait_cycles[i + 2]
            start_pos = segment_data[start, 27]
            end_pos = segment_data[end, 27]
            distance = np.linalg.norm(end_pos - start_pos)
            stride_lengths.append(distance)
        return np.mean(stride_lengths) if stride_lengths else 0

    def calculate_walking_speed(self, cadence, stride_length):
        return (cadence * stride_length) / 60  # m/s

    def calculate_joint_angles(self, segment_data):
        hip_angle = self.calculate_angle(23, 25, 27, segment_data)
        knee_angle = self.calculate_angle(25, 27, 31, segment_data)
        ankle_angle = self.calculate_angle(25, 27, 29, segment_data)
        return hip_angle, knee_angle, ankle_angle

    def calculate_angle(self, joint1, joint2, joint3, segment_data):
        v1 = segment_data[:, joint1] - segment_data[:, joint2]
        v2 = segment_data[:, joint3] - segment_data[:, joint2]
        v1_u = v1 / np.linalg.norm(v1, axis=1)[:, np.newaxis]
        v2_u = v2 / np.linalg.norm(v2, axis=1)[:, np.newaxis]
        return np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0))

    def analyze_video(self, video_data):
        self.pose_data = video_data
        results = []
        segment_start = 0
        
        for i in range(1, len(self.pose_data)):
            if self.is_direction_change(i):
                if i - segment_start > self.fps * 2:  # 최소 2초
                    segment_result = self.analyze_segment(segment_start, i)
                    results.append(segment_result)
                segment_start = i

        if len(self.pose_data) - segment_start > self.fps * 2:
            segment_result = self.analyze_segment(segment_start, len(self.pose_data))
            results.append(segment_result)

        return results

    def is_direction_change(self, index):
        prev_hip = (self.pose_data[index-1, 23] + self.pose_data[index-1, 24]) / 2
        curr_hip = (self.pose_data[index, 23] + self.pose_data[index, 24]) / 2
        return np.sign(curr_hip[2] - prev_hip[2]) != np.sign(prev_hip[2] - self.pose_data[max(0, index-2), 23][2])

    def analyze_segment(self, start, end):
        segment_data = self.pose_data[start:end]
        left_ankle_y = segment_data[:, 27, 1]
        peaks, _ = find_peaks(left_ankle_y, distance=15)

        cadence = self.calculate_cadence(peaks, (end - start) / self.fps)
        stride_length = self.calculate_stride_length(peaks, segment_data)
        walking_speed = self.calculate_walking_speed(cadence, stride_length)
        hip_angle, knee_angle, ankle_angle = self.calculate_joint_angles(segment_data)

        return {
            "segment": (start, end),
            "cadence": cadence,
            "stride_length": stride_length,
            "walking_speed": walking_speed,
            "hip_angle": hip_angle.tolist(),
            "knee_angle": knee_angle.tolist(),
            "ankle_angle": ankle_angle.tolist()
        }

    def save_results_to_csv(self, results, output_file):
        all_data = []
        for i, result in enumerate(results):
            segment_data = {
                'segment': i + 1,
                'start_frame': result['segment'][0],
                'end_frame': result['segment'][1],
                'cadence': result['cadence'],
                'stride_length': result['stride_length'],
                'walking_speed': result['walking_speed'],
            }
            all_data.append(segment_data)

            for joint in ['hip', 'knee', 'ankle']:
                angle_data = result[f'{joint}_angle']
                for j, angle in enumerate(angle_data):
                    frame_data = segment_data.copy()
                    frame_data.update({
                        'frame': result['segment'][0] + j,
                        f'{joint}_angle': angle
                    })
                    all_data.append(frame_data)

        df_results = pd.DataFrame(all_data)
        df_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
