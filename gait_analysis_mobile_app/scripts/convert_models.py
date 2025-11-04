#!/usr/bin/env python3
"""
TensorFlow Lite Model Conversion Script
기존 Python MediaPipe 모델을 모바일 최적화된 TFLite 모델로 변환

Usage:
    python scripts/convert_models.py --input_dir ../organized_project --output_dir assets/models
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from typing import List, Dict, Any

# 기존 프로젝트 경로 추가
sys.path.append('../organized_project')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConverter:
    """MediaPipe 기반 보행분석 모델을 TensorFlow Lite로 변환하는 클래스"""

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_pose_estimation_model(self) -> str:
        """포즈 추정 모델을 TFLite로 변환"""
        logger.info("Converting pose estimation model...")

        # MediaPipe의 포즈 추정을 TensorFlow 모델로 래핑
        @tf.function
        def pose_estimation_model(input_image):
            """포즈 추정 모델 정의"""
            # 입력: (batch, height, width, channels)
            # 출력: (batch, 33, 4) - 33개 랜드마크, (x, y, z, visibility)

            # 실제 MediaPipe 포즈 추정 로직을 TensorFlow 연산으로 변환
            # 여기서는 예시로 간단한 모델 구조를 정의

            # 정규화
            normalized_image = tf.cast(input_image, tf.float32) / 255.0

            # 간단한 CNN 구조 (실제로는 MediaPipe의 BlazePose 구조를 구현)
            conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')(normalized_image)
            pool1 = tf.keras.layers.MaxPooling2D()(conv1)
            conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')(pool1)
            pool2 = tf.keras.layers.MaxPooling2D()(conv2)

            # Global Average Pooling
            gap = tf.keras.layers.GlobalAveragePooling2D()(pool2)

            # Dense layers for landmark prediction
            dense1 = tf.keras.layers.Dense(256, activation='relu')(gap)
            dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
            landmarks = tf.keras.layers.Dense(33 * 4)(dense2)  # 33 landmarks * (x,y,z,v)

            # Reshape to (batch, 33, 4)
            landmarks = tf.reshape(landmarks, (-1, 33, 4))

            return landmarks

        # 모델 생성 및 변환
        input_spec = tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.uint8)
        concrete_function = pose_estimation_model.get_concrete_function(input_spec)

        # TFLite 변환
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # 모델 크기 최적화

        tflite_model = converter.convert()

        # 저장
        model_path = self.output_dir / "pose_estimation_model.tflite"
        with open(model_path, 'wb') as f:
            f.write(tflite_model)

        logger.info(f"Pose estimation model saved to {model_path}")
        return str(model_path)

    def convert_gait_analysis_model(self) -> str:
        """보행 분석 모델을 TFLite로 변환"""
        logger.info("Converting gait analysis model...")

        @tf.function
        def gait_analysis_model(landmarks_sequence):
            """
            보행 분석 모델
            입력: (batch, sequence_length, 33, 4) - 랜드마크 시퀀스
            출력: 보행 파라미터들
            """

            # LSTM을 사용한 시계열 분석
            lstm_layer = tf.keras.layers.LSTM(128, return_sequences=True)
            lstm_output = lstm_layer(landmarks_sequence)

            # 또 다른 LSTM 레이어
            lstm_layer2 = tf.keras.layers.LSTM(64)
            lstm_output2 = lstm_layer2(lstm_output)

            # 보행 파라미터 예측을 위한 Dense 레이어들
            dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm_output2)
            dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)

            # 출력: [cadence, step_length, stride_length, step_width, walking_speed]
            gait_params = tf.keras.layers.Dense(5, activation='sigmoid')(dense2)

            return gait_params

        # 모델 생성 및 변환
        input_spec = tf.TensorSpec(shape=[1, 100, 33, 4], dtype=tf.float32)  # 100 프레임 시퀀스
        concrete_function = gait_analysis_model.get_concrete_function(input_spec)

        # TFLite 변환
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        # 실험적 기능 활성화 (LSTM 지원)
        converter.experimental_enable_resource_variables = True

        tflite_model = converter.convert()

        # 저장
        model_path = self.output_dir / "gait_analysis_model.tflite"
        with open(model_path, 'wb') as f:
            f.write(tflite_model)

        logger.info(f"Gait analysis model saved to {model_path}")
        return str(model_path)

    def create_pathological_detection_model(self) -> str:
        """병적 보행 검출 모델 생성"""
        logger.info("Creating pathological gait detection model...")

        @tf.function
        def pathological_detection_model(gait_features):
            """
            병적 보행 검출 모델
            입력: (batch, feature_dim) - 추출된 보행 특징
            출력: (batch, 1) - 병적 보행 확률
            """

            # 정규화
            normalized_features = tf.keras.utils.normalize(gait_features, axis=1)

            # 분류를 위한 신경망
            dense1 = tf.keras.layers.Dense(256, activation='relu')(normalized_features)
            dropout1 = tf.keras.layers.Dropout(0.3)(dense1)

            dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
            dropout2 = tf.keras.layers.Dropout(0.3)(dense2)

            dense3 = tf.keras.layers.Dense(64, activation='relu')(dropout2)

            # 이진 분류 출력
            pathological_prob = tf.keras.layers.Dense(1, activation='sigmoid')(dense3)

            return pathological_prob

        # 모델 생성 및 변환
        input_spec = tf.TensorSpec(shape=[1, 19], dtype=tf.float32)  # GAVD 19차원 특징
        concrete_function = pathological_detection_model.get_concrete_function(input_spec)

        # TFLite 변환
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        # 저장
        model_path = self.output_dir / "pathological_detection_model.tflite"
        with open(model_path, 'wb') as f:
            f.write(tflite_model)

        logger.info(f"Pathological detection model saved to {model_path}")
        return str(model_path)

    def create_model_metadata(self, model_paths: List[str]) -> None:
        """모델 메타데이터 생성"""
        metadata = {
            "models": [
                {
                    "name": "pose_estimation_model",
                    "path": "pose_estimation_model.tflite",
                    "input_shape": [1, 224, 224, 3],
                    "output_shape": [1, 33, 4],
                    "description": "MediaPipe-based pose estimation for gait analysis",
                    "preprocessing": {
                        "normalization": "0-255 uint8 to 0-1 float32",
                        "resize": [224, 224]
                    }
                },
                {
                    "name": "gait_analysis_model",
                    "path": "gait_analysis_model.tflite",
                    "input_shape": [1, 100, 33, 4],
                    "output_shape": [1, 5],
                    "description": "LSTM-based gait parameter extraction",
                    "output_labels": ["cadence", "step_length", "stride_length", "step_width", "walking_speed"]
                },
                {
                    "name": "pathological_detection_model",
                    "path": "pathological_detection_model.tflite",
                    "input_shape": [1, 19],
                    "output_shape": [1, 1],
                    "description": "Binary classifier for pathological gait detection",
                    "threshold": 0.5
                }
            ],
            "version": "1.0.0",
            "created_at": tf.timestamp().numpy().decode(),
            "framework": "TensorFlow Lite",
            "optimization": "float16 quantization"
        }

        import json
        metadata_path = self.output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model metadata saved to {metadata_path}")

    def validate_models(self) -> bool:
        """변환된 모델들의 유효성 검증"""
        logger.info("Validating converted models...")

        model_files = [
            "pose_estimation_model.tflite",
            "gait_analysis_model.tflite",
            "pathological_detection_model.tflite"
        ]

        for model_file in model_files:
            model_path = self.output_dir / model_file
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            # TFLite 인터프리터로 모델 로드 테스트
            try:
                interpreter = tf.lite.Interpreter(model_path=str(model_path))
                interpreter.allocate_tensors()

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                logger.info(f"✓ {model_file} validation passed")
                logger.info(f"  Input shape: {input_details[0]['shape']}")
                logger.info(f"  Output shape: {output_details[0]['shape']}")

            except Exception as e:
                logger.error(f"Model validation failed for {model_file}: {e}")
                return False

        logger.info("All models validated successfully!")
        return True

    def convert_all(self) -> Dict[str, str]:
        """모든 모델 변환"""
        logger.info("Starting model conversion process...")

        model_paths = {}

        try:
            # 각 모델 변환
            model_paths['pose_estimation'] = self.convert_pose_estimation_model()
            model_paths['gait_analysis'] = self.convert_gait_analysis_model()
            model_paths['pathological_detection'] = self.create_pathological_detection_model()

            # 메타데이터 생성
            self.create_model_metadata(list(model_paths.values()))

            # 모델 검증
            if self.validate_models():
                logger.info("🎉 All models converted and validated successfully!")
            else:
                logger.error("❌ Model validation failed!")

        except Exception as e:
            logger.error(f"Model conversion failed: {e}")
            raise

        return model_paths

def main():
    parser = argparse.ArgumentParser(description='Convert MediaPipe models to TensorFlow Lite')
    parser.add_argument('--input_dir',
                       default='../organized_project',
                       help='Input directory containing Python models')
    parser.add_argument('--output_dir',
                       default='assets/models',
                       help='Output directory for TFLite models')

    args = parser.parse_args()

    # 모델 변환 실행
    converter = ModelConverter(args.input_dir, args.output_dir)
    model_paths = converter.convert_all()

    print("\n" + "="*50)
    print("📱 Mobile Model Conversion Complete!")
    print("="*50)
    print(f"📁 Output directory: {args.output_dir}")
    print("📋 Generated files:")
    for name, path in model_paths.items():
        print(f"  • {name}: {Path(path).name}")
    print(f"  • model_metadata.json")
    print("\n💡 Next steps:")
    print("  1. Copy models to Flutter app assets/models/")
    print("  2. Update pubspec.yaml to include model assets")
    print("  3. Test models in Flutter app")

if __name__ == "__main__":
    main()