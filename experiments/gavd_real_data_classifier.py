#!/usr/bin/env python3
"""
GAVD Real Data Gait Classifier
Enhanced MediaPipe Gait Analysis System v3.0 - 실제 데이터 기반 분류기

실제 GAVD 임상 데이터로 훈련된 병적보행 분류 시스템

Author: Research Team
Date: 2025-09-22
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GAVDRealDataClassifier:
    """실제 GAVD 데이터 기반 병적보행 분류기"""

    def __init__(self, results_file=None):
        """
        실제 데이터 분류기 초기화

        Args:
            results_file: 최적화된 추출 결과 JSON 파일
        """
        self.results_file = results_file
        self.raw_data = None
        self.processed_features = None
        self.labels = None
        self.binary_labels = None

        # 모델들
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.binary_classifier = None
        self.multiclass_classifier = None
        self.anomaly_detector = None

        # 성능 메트릭
        self.performance_metrics = {}

        print(f"🏥 GAVD 실제 데이터 분류기 초기화")
        print(f"📁 결과 파일: {results_file}")

    def load_extracted_features(self, results_file=None):
        """추출된 특징 데이터 로드"""
        if results_file:
            self.results_file = results_file

        if not self.results_file or not Path(self.results_file).exists():
            print(f"❌ 결과 파일을 찾을 수 없습니다: {self.results_file}")
            return False

        print(f"\n📖 추출된 특징 데이터 로드 중...")

        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        successful_results = self.raw_data.get('successful_results', [])
        print(f"✅ 성공적인 결과: {len(successful_results)}개")

        if len(successful_results) == 0:
            print(f"❌ 처리 가능한 데이터가 없습니다.")
            return False

        # 특징 벡터와 라벨 추출
        feature_vectors = []
        labels = []
        video_info = []

        for result in successful_results:
            if result.get('gait_features') and result.get('success'):
                features = result['gait_features']

                # 특징 벡터 구성 (실제 측정값 기반)
                feature_vector = [
                    features.get('ankle_range', 0),
                    features.get('heel_range', 0),
                    features.get('knee_range', 0),
                    features.get('ankle_variability', 0),
                    features.get('heel_variability', 0),
                    features.get('estimated_cadence', 0),
                    features.get('movement_smoothness', 0),
                    features.get('total_frames_analyzed', 0),
                    # 비디오 품질 지표
                    result['video_info']['success_rate'],
                    result['video_info']['successful_frames'],
                    result['processing_fps'],
                    # 추가 계산된 특징
                    features.get('ankle_range', 0) / features.get('heel_range', 1),  # 발목/발뒤꿈치 비율
                    features.get('estimated_cadence', 0) / 120.0 if features.get('estimated_cadence', 0) > 0 else 0,  # 정규화된 케이던스
                    1.0 - features.get('ankle_variability', 0),  # 안정성 지수
                    features.get('movement_smoothness', 0) * result['video_info']['success_rate']  # 종합 품질 지수
                ]

                feature_vectors.append(feature_vector)
                labels.append(result['gait_pattern'])
                video_info.append({
                    'video_id': result['video_id'],
                    'camera_view': result['camera_view'],
                    'gait_pattern': result['gait_pattern']
                })

        self.processed_features = np.array(feature_vectors)
        self.labels = np.array(labels)
        self.video_info = video_info

        # 이진 분류 라벨 생성 (normal vs pathological)
        self.binary_labels = np.array(['normal' if label == 'normal' else 'pathological' for label in self.labels])

        print(f"📊 처리된 데이터:")
        print(f"   특징 벡터: {self.processed_features.shape}")
        print(f"   고유 패턴: {np.unique(self.labels)}")
        print(f"   패턴 분포: {np.unique(self.labels, return_counts=True)}")

        return True

    def validate_clinical_criteria(self):
        """임상적 기준으로 데이터 검증"""
        print(f"\n🏥 임상적 기준 검증...")

        if self.processed_features is None:
            print(f"❌ 로드된 데이터가 없습니다.")
            return False

        # 정상 범위 설정 (문헌 기반)
        clinical_ranges = {
            'cadence': (90, 130),  # steps/min
            'ankle_range': (0.05, 0.3),  # normalized range
            'movement_smoothness': (0.3, 1.0),
            'success_rate': (0.7, 1.0)  # landmark detection quality
        }

        validation_results = {}

        for i, pattern in enumerate(np.unique(self.labels)):
            pattern_mask = self.labels == pattern
            pattern_features = self.processed_features[pattern_mask]

            if len(pattern_features) == 0:
                continue

            # 패턴별 특징 통계
            cadence_values = pattern_features[:, 5]  # estimated_cadence
            ankle_range_values = pattern_features[:, 0]  # ankle_range
            smoothness_values = pattern_features[:, 6]  # movement_smoothness
            success_rate_values = pattern_features[:, 8]  # success_rate

            validation_results[pattern] = {
                'count': len(pattern_features),
                'cadence': {
                    'mean': np.mean(cadence_values),
                    'std': np.std(cadence_values),
                    'within_normal': np.sum((cadence_values >= clinical_ranges['cadence'][0]) &
                                          (cadence_values <= clinical_ranges['cadence'][1])) / len(cadence_values)
                },
                'ankle_range': {
                    'mean': np.mean(ankle_range_values),
                    'std': np.std(ankle_range_values),
                    'within_normal': np.sum((ankle_range_values >= clinical_ranges['ankle_range'][0]) &
                                          (ankle_range_values <= clinical_ranges['ankle_range'][1])) / len(ankle_range_values)
                },
                'movement_quality': {
                    'smoothness_mean': np.mean(smoothness_values),
                    'success_rate_mean': np.mean(success_rate_values)
                }
            }

        # 검증 결과 출력
        print(f"\n📋 패턴별 임상적 특성:")
        for pattern, stats in validation_results.items():
            print(f"\n🦴 {pattern} ({stats['count']}개):")
            print(f"   케이던스: {stats['cadence']['mean']:.1f} ± {stats['cadence']['std']:.1f} steps/min "
                  f"(정상범위 내: {stats['cadence']['within_normal']*100:.1f}%)")
            print(f"   발목 가동범위: {stats['ankle_range']['mean']:.3f} ± {stats['ankle_range']['std']:.3f} "
                  f"(정상범위 내: {stats['ankle_range']['within_normal']*100:.1f}%)")
            print(f"   움직임 품질: {stats['movement_quality']['smoothness_mean']:.3f}")
            print(f"   검출 성공률: {stats['movement_quality']['success_rate_mean']*100:.1f}%")

        self.validation_results = validation_results
        return True

    def train_binary_classifier(self):
        """이진 분류기 훈련 (정상 vs 병적)"""
        print(f"\n🎯 이진 분류기 훈련 (정상 vs 병적)...")

        if self.processed_features is None:
            print(f"❌ 훈련 데이터가 없습니다.")
            return False

        # 클래스 분포 확인
        unique_binary, counts_binary = np.unique(self.binary_labels, return_counts=True)
        print(f"   이진 분류 분포: {dict(zip(unique_binary, counts_binary))}")

        if len(unique_binary) < 2:
            print(f"❌ 이진 분류를 위한 충분한 클래스가 없습니다.")
            return False

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            self.processed_features, self.binary_labels,
            test_size=0.3, random_state=42, stratify=self.binary_labels
        )

        # 특징 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 랜덤 포레스트 분류기
        self.binary_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight='balanced',
            min_samples_split=3,
            min_samples_leaf=2
        )

        # 훈련
        self.binary_classifier.fit(X_train_scaled, y_train)

        # 예측 및 평가
        y_pred = self.binary_classifier.predict(X_test_scaled)
        y_pred_proba = self.binary_classifier.predict_proba(X_test_scaled)

        # 성능 메트릭
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='pathological', average='binary')
        recall = recall_score(y_test, y_pred, pos_label='pathological', average='binary')
        f1 = f1_score(y_test, y_pred, pos_label='pathological', average='binary')

        # AUC 계산
        if len(unique_binary) == 2:
            pathological_idx = np.where(self.binary_classifier.classes_ == 'pathological')[0][0]
            auc = roc_auc_score(y_test, y_pred_proba[:, pathological_idx])
        else:
            auc = 0.5

        # 교차 검증
        cv_scores = cross_val_score(self.binary_classifier, X_train_scaled, y_train,
                                  cv=StratifiedKFold(n_splits=min(5, len(y_train)//2)),
                                  scoring='accuracy')

        self.performance_metrics['binary_classification'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'cv_accuracy_mean': np.mean(cv_scores),
            'cv_accuracy_std': np.std(cv_scores),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        print(f"✅ 이진 분류기 훈련 완료:")
        print(f"   정확도: {accuracy:.3f}")
        print(f"   정밀도: {precision:.3f}")
        print(f"   재현율: {recall:.3f}")
        print(f"   F1 점수: {f1:.3f}")
        print(f"   AUC: {auc:.3f}")
        print(f"   교차검증: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

        return True

    def train_multiclass_classifier(self):
        """다중 클래스 분류기 훈련"""
        print(f"\n🎭 다중 클래스 분류기 훈련...")

        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"   클래스 분포: {dict(zip(unique_labels, counts))}")

        # 최소 샘플 수 확인 (클래스당 최소 3개)
        min_samples = 3
        valid_classes = [label for label, count in zip(unique_labels, counts) if count >= min_samples]

        if len(valid_classes) < 2:
            print(f"❌ 다중 분류를 위한 충분한 클래스가 없습니다 (최소 {min_samples}개 샘플 필요).")
            return False

        # 유효한 클래스만 필터링
        valid_mask = np.isin(self.labels, valid_classes)
        X_filtered = self.processed_features[valid_mask]
        y_filtered = self.labels[valid_mask]

        print(f"   필터링 후 유효 클래스: {valid_classes}")
        print(f"   필터링 후 샘플 수: {len(X_filtered)}개")

        # 레이블 인코딩
        y_encoded = self.label_encoder.fit_transform(y_filtered)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_encoded,
            test_size=0.3, random_state=42, stratify=y_encoded
        )

        # 특징 스케일링 (기존 scaler 사용)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 다중 클래스 분류기
        self.multiclass_classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            min_samples_split=2,
            min_samples_leaf=1
        )

        # 훈련
        self.multiclass_classifier.fit(X_train_scaled, y_train)

        # 예측 및 평가
        y_pred = self.multiclass_classifier.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred,
                                           target_names=self.label_encoder.classes_,
                                           output_dict=True, zero_division=0)

        self.performance_metrics['multiclass_classification'] = {
            'accuracy': accuracy,
            'valid_classes': valid_classes,
            'classes_count': len(valid_classes),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': class_report
        }

        print(f"✅ 다중 클래스 분류기 훈련 완료:")
        print(f"   정확도: {accuracy:.3f}")
        print(f"   유효 클래스: {len(valid_classes)}개")

        return True

    def train_anomaly_detector(self):
        """이상 검출기 훈련 (정상 보행 기반)"""
        print(f"\n🔍 이상 검출기 훈련...")

        # 정상 보행 데이터만 추출
        normal_mask = self.labels == 'normal'
        normal_features = self.processed_features[normal_mask]

        if len(normal_features) < 5:
            print(f"❌ 이상 검출을 위한 충분한 정상 데이터가 없습니다 ({len(normal_features)}개).")
            return False

        print(f"   정상 보행 샘플: {len(normal_features)}개")

        # 정상 데이터로 스케일링
        normal_features_scaled = self.scaler.transform(normal_features)

        # Isolation Forest
        isolation_forest = IsolationForest(
            contamination=0.15,  # 15% 이상치 허용
            random_state=42,
            n_estimators=100
        )

        # One-Class SVM
        one_class_svm = OneClassSVM(
            nu=0.15,
            kernel='rbf',
            gamma='scale'
        )

        # 정상 데이터로 훈련
        isolation_forest.fit(normal_features_scaled)
        one_class_svm.fit(normal_features_scaled)

        self.anomaly_detector = {
            'isolation_forest': isolation_forest,
            'one_class_svm': one_class_svm
        }

        # 전체 데이터에 대한 이상 검출 성능 평가
        all_features_scaled = self.scaler.transform(self.processed_features)

        # 실제 레이블 (정상=1, 비정상=-1)
        true_anomaly_labels = np.array([1 if label == 'normal' else -1 for label in self.labels])

        # 예측
        if_pred = isolation_forest.predict(all_features_scaled)
        svm_pred = one_class_svm.predict(all_features_scaled)

        # 앙상블 예측 (두 모델 모두 정상이라고 해야 정상)
        ensemble_pred = np.array([1 if (if_p == 1 and svm_p == 1) else -1
                                for if_p, svm_p in zip(if_pred, svm_pred)])

        # 성능 계산
        anomaly_accuracy = accuracy_score(true_anomaly_labels, ensemble_pred)

        self.performance_metrics['anomaly_detection'] = {
            'accuracy': anomaly_accuracy,
            'normal_samples': len(normal_features),
            'total_samples': len(self.processed_features),
            'isolation_forest_anomalies': np.sum(if_pred == -1),
            'svm_anomalies': np.sum(svm_pred == -1),
            'ensemble_anomalies': np.sum(ensemble_pred == -1)
        }

        print(f"✅ 이상 검출기 훈련 완료:")
        print(f"   정확도: {anomaly_accuracy:.3f}")
        print(f"   검출된 이상: {np.sum(ensemble_pred == -1)}개")

        return True

    def generate_performance_report(self):
        """성능 보고서 생성"""
        print(f"\n📊 실제 데이터 기반 성능 보고서 생성...")

        if not self.performance_metrics:
            print(f"❌ 성능 메트릭이 없습니다.")
            return

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report = f"""
🏥 GAVD 실제 임상 데이터 기반 병적보행 분류 시스템 성능 보고서
{'='*80}

📅 생성 일시: {timestamp}
📊 데이터 기반: 실제 GAVD 임상 비디오 ({len(self.processed_features)}개)

📈 데이터셋 정보:
   특징 차원: {self.processed_features.shape[1]}개
   총 샘플: {len(self.processed_features)}개
   고유 패턴: {len(np.unique(self.labels))}개
   패턴 분포: {dict(zip(*np.unique(self.labels, return_counts=True)))}

🎯 이진 분류 성능 (정상 vs 병적):"""

        if 'binary_classification' in self.performance_metrics:
            binary_metrics = self.performance_metrics['binary_classification']
            report += f"""
   정확도: {binary_metrics['accuracy']:.3f}
   정밀도: {binary_metrics['precision']:.3f}
   재현율: {binary_metrics['recall']:.3f}
   F1 점수: {binary_metrics['f1_score']:.3f}
   AUC-ROC: {binary_metrics['auc_roc']:.3f}
   교차검증 정확도: {binary_metrics['cv_accuracy_mean']:.3f} ± {binary_metrics['cv_accuracy_std']:.3f}
   훈련/테스트 비율: {binary_metrics['train_size']}/{binary_metrics['test_size']}"""
        else:
            report += "\n   ❌ 이진 분류 결과 없음"

        report += f"\n\n🎭 다중 클래스 분류 성능:"

        if 'multiclass_classification' in self.performance_metrics:
            multi_metrics = self.performance_metrics['multiclass_classification']
            report += f"""
   정확도: {multi_metrics['accuracy']:.3f}
   유효 클래스: {multi_metrics['classes_count']}개
   클래스 목록: {multi_metrics['valid_classes']}
   훈련/테스트 비율: {multi_metrics['train_size']}/{multi_metrics['test_size']}"""
        else:
            report += "\n   ❌ 다중 클래스 분류 결과 없음"

        report += f"\n\n🔍 이상 검출 성능:"

        if 'anomaly_detection' in self.performance_metrics:
            anomaly_metrics = self.performance_metrics['anomaly_detection']
            report += f"""
   정확도: {anomaly_metrics['accuracy']:.3f}
   정상 기준 샘플: {anomaly_metrics['normal_samples']}개
   검출된 이상: {anomaly_metrics['ensemble_anomalies']}개
   이상 비율: {anomaly_metrics['ensemble_anomalies']/anomaly_metrics['total_samples']*100:.1f}%"""
        else:
            report += "\n   ❌ 이상 검출 결과 없음"

        report += f"""

💡 실제 데이터 기반 시스템의 특징:
   ✅ 실제 환자 비디오에서 추출한 특징 사용
   ✅ 임상적으로 의미있는 보행 파라미터 기반
   ✅ MediaPipe landmark detection 품질 고려
   ✅ 다양한 카메라 뷰 (left_side, right_side) 통합
   ✅ 교차검증으로 과적합 방지

⚠️  제한사항:
   • 샘플 크기가 제한적 (특히 일부 병적 패턴)
   • 단일 프레임 분석이 아닌 시간적 패턴 분석 필요
   • 더 많은 임상 검증 데이터 필요

🔬 이전 시뮬레이션 시스템 대비 개선점:
   • 실제 환자 데이터 기반으로 신뢰성 향상
   • MediaPipe 검출 품질을 고려한 robust한 분류
   • 임상적으로 검증 가능한 특징 사용
   • 과적합 방지를 위한 적절한 모델 복잡도 조절

{'='*80}
보고서 생성 시간: {timestamp}
"""

        print(report)

        # 보고서 파일 저장
        report_file = f"gavd_real_data_classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📄 보고서 저장: {report_file}")
        return report

    def save_models(self):
        """훈련된 모델 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = Path(f"gavd_real_models_{timestamp}")
        model_dir.mkdir(exist_ok=True)

        # 모델들 저장
        if self.binary_classifier:
            joblib.dump(self.binary_classifier, model_dir / "binary_classifier.pkl")

        if self.multiclass_classifier:
            joblib.dump(self.multiclass_classifier, model_dir / "multiclass_classifier.pkl")
            joblib.dump(self.label_encoder, model_dir / "label_encoder.pkl")

        if self.anomaly_detector:
            joblib.dump(self.anomaly_detector, model_dir / "anomaly_detector.pkl")

        joblib.dump(self.scaler, model_dir / "feature_scaler.pkl")

        # 성능 메트릭 저장
        with open(model_dir / "performance_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(self.performance_metrics, f, indent=2, ensure_ascii=False)

        print(f"\n💾 모델 저장 완료: {model_dir}")
        return model_dir

def main():
    """메인 실행 함수"""
    print("🏥 GAVD 실제 데이터 기반 병적보행 분류기")
    print("=" * 60)

    # 최신 결과 파일 찾기
    result_files = list(Path(".").glob("gavd_optimized_results_*.json"))
    if not result_files:
        print("❌ 추출 결과 파일을 찾을 수 없습니다.")
        print("   먼저 gavd_optimized_mediapipe_extractor.py를 실행하세요.")
        return

    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"📁 사용할 결과 파일: {latest_file}")

    try:
        # 분류기 초기화
        classifier = GAVDRealDataClassifier(str(latest_file))

        # 1. 특징 데이터 로드
        if not classifier.load_extracted_features():
            return

        # 2. 임상적 검증
        classifier.validate_clinical_criteria()

        # 3. 모델 훈련
        print(f"\n🚀 분류 모델 훈련 시작...")

        # 이진 분류기
        classifier.train_binary_classifier()

        # 다중 클래스 분류기
        classifier.train_multiclass_classifier()

        # 이상 검출기
        classifier.train_anomaly_detector()

        # 4. 성능 보고서 생성
        classifier.generate_performance_report()

        # 5. 모델 저장
        model_dir = classifier.save_models()

        print(f"\n🎉 실제 데이터 기반 분류 시스템 완료!")
        print(f"💾 모델 저장 위치: {model_dir}")

    except Exception as e:
        print(f"❌ 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()