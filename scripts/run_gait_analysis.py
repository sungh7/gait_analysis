#!/usr/bin/env python3
"""
Enhanced MediaPipe Gait Analysis System - Master Execution Script
Version 2.0 - 통합 실행 스크립트

사용법:
    python3 run_gait_analysis.py [옵션]

옵션:
    --quick-test        : 3명 피험자 빠른 테스트 (기본값)
    --full-validation   : 전체 21명 피험자 대규모 검증
    --single-subject N  : 특정 피험자 N만 분석
    --clinical-only     : 임상 최적화만 수행
    --help             : 도움말 표시

Author: Research Team
Date: 2025-09-22
"""

import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

# 핵심 모듈 임포트
from batch_validation_system import BatchValidationSystem
from clinical_optimization_system import ClinicalOptimizationSystem
from advanced_validation_framework import AdvancedValidationFramework

def print_banner():
    """시스템 배너 출력"""
    print("=" * 80)
    print("🏥 Enhanced MediaPipe Gait Analysis System v2.0")
    print("=" * 80)
    print("📊 Features: Physical Calibration + DTW/SPM Validation + Clinical Optimization")
    print("🔬 Research Team - September 2025")
    print("=" * 80)

def run_quick_test():
    """빠른 테스트 (3명 피험자)"""
    print("\n🚀 빠른 테스트 모드 - 3명 피험자 분석")
    print("-" * 60)

    # 시스템 초기화
    batch_system = BatchValidationSystem(max_workers=4)
    clinical_optimizer = ClinicalOptimizationSystem()

    # 피험자 선택
    subjects = batch_system.discover_subjects()
    test_subjects = subjects[:3]

    print(f"📋 선택된 피험자: {len(test_subjects)}명")
    for subject in test_subjects:
        print(f"   • 피험자 {subject['id']}: {Path(subject['sagittal_video']).name}")

    # 분석 실행
    print(f"\n🔬 분석 시작... (프레임 제한: 300)")
    start_time = time.time()

    results = batch_system.run_batch_validation(test_subjects, frame_limit=300)

    # 임상 최적화
    print(f"\n🏥 임상 최적화 수행...")
    optimization = clinical_optimizer.optimize_processing_parameters(results)
    clinical_report = clinical_optimizer.generate_clinical_report(
        results, results.get('validation_results', {})
    )

    # 결과 출력
    processing_time = time.time() - start_time
    print(f"\n📊 빠른 테스트 완료! (소요시간: {processing_time:.1f}초)")
    print(f"=" * 60)

    # 성능 요약
    mp_results = results.get('mediapipe_results', [])
    if mp_results:
        avg_success_rate = sum(r['processing_stats']['success_rate'] for r in mp_results) / len(mp_results)
        print(f"🎬 MediaPipe 처리 성공률: {avg_success_rate:.1%}")

    # 고급 검증 결과
    advanced_results = results.get('validation_results', {}).get('advanced_results', {})
    if advanced_results and 'summary' in advanced_results:
        summary = advanced_results['summary']
        print(f"📈 DTW 평균 점수: {summary.get('avg_dtw_score', 0):.3f}")
        print(f"📊 SPM 평균 점수: {summary.get('avg_spm_score', 0):.3f}")

    # 임상 권장사항
    frame_opt = optimization.get('frame_rate_optimization', {})
    print(f"⚙️  권장 FPS: {frame_opt.get('recommended_fps', 30)}")

    grade = clinical_report['performance_assessment'].get('overall_grade', 'N/A')
    print(f"🏆 시스템 등급: {grade}")

    print(f"\n💾 결과 저장: quick_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    clinical_optimizer.save_clinical_report(
        clinical_report,
        f"quick_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

def run_full_validation():
    """전체 검증 (21명 피험자)"""
    print("\n🔬 전체 검증 모드 - 21명 피험자 대규모 분석")
    print("-" * 60)
    print("⚠️  주의: 이 분석은 30-60분 소요될 수 있습니다.")

    confirm = input("계속하시겠습니까? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ 전체 검증이 취소되었습니다.")
        return

    # 시스템 초기화
    batch_system = BatchValidationSystem(max_workers=8)
    clinical_optimizer = ClinicalOptimizationSystem()

    # 모든 피험자 분석
    subjects = batch_system.discover_subjects()

    print(f"📋 분석 대상: {len(subjects)}명 피험자")
    print(f"🔬 분석 시작... (프레임 제한: 800)")

    start_time = time.time()
    results = batch_system.run_batch_validation(subjects, frame_limit=99999)

    # 임상 최적화
    print(f"\n🏥 임상 최적화 및 보고서 생성...")
    optimization = clinical_optimizer.optimize_processing_parameters(results)
    clinical_report = clinical_optimizer.generate_clinical_report(
        results, results.get('validation_results', {})
    )

    processing_time = time.time() - start_time
    print(f"\n🎉 전체 검증 완료! (총 소요시간: {processing_time/60:.1f}분)")

    # 상세 결과 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    clinical_optimizer.save_clinical_report(
        clinical_report,
        f"full_validation_report_{timestamp}.json"
    )

    print(f"💾 상세 보고서 저장: full_validation_report_{timestamp}.json")

def run_single_subject(subject_id):
    """단일 피험자 분석"""
    print(f"\n👤 단일 피험자 분석 - 피험자 {subject_id}")
    print("-" * 60)

    # 시스템 초기화
    batch_system = BatchValidationSystem(max_workers=2)

    # 특정 피험자 선택
    subjects = batch_system.discover_subjects()
    target_subject = None

    for subject in subjects:
        if subject['id'] == subject_id:
            target_subject = subject
            break

    if not target_subject:
        print(f"❌ 피험자 {subject_id}를 찾을 수 없습니다.")
        print(f"📋 사용 가능한 피험자: {[s['id'] for s in subjects[:10]]}...")
        return

    print(f"📹 비디오: {Path(target_subject['sagittal_video']).name}")

    # 분석 실행
    results = batch_system.run_batch_validation([target_subject], frame_limit=600)

    # 결과 출력
    mp_result = results.get('mediapipe_results', [])
    if mp_result:
        result = mp_result[0]
        stats = result['processing_stats']
        gait_params = result['gait_parameters']

        print(f"\n📊 분석 결과:")
        print(f"   성공률: {stats['success_rate']:.1%}")
        print(f"   처리 시간: {stats['processing_time']:.1f}초")
        print(f"   Cadence: {gait_params.get('cadence', 0):.1f} steps/min")
        print(f"   보행 속도: {gait_params.get('walking_speed', 0):.2f} m/s")
        print(f"   스텝 길이: {gait_params.get('step_length_left', 0):.3f} m")

    print(f"\n💾 결과 저장: subject_{subject_id}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

def run_clinical_optimization():
    """임상 최적화만 수행"""
    print("\n🏥 임상 최적화 전용 모드")
    print("-" * 60)

    # 기존 결과 파일 찾기
    result_files = list(Path(".").glob("batch_validation_results_*.json"))

    if not result_files:
        print("❌ 기존 분석 결과 파일을 찾을 수 없습니다.")
        print("💡 먼저 --quick-test 또는 --full-validation을 실행하세요.")
        return

    # 가장 최신 파일 사용
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"📂 분석 결과 로드: {latest_file.name}")

    import json
    with open(latest_file, 'r') as f:
        results = json.load(f)

    # 임상 최적화 수행
    clinical_optimizer = ClinicalOptimizationSystem()
    optimization = clinical_optimizer.optimize_processing_parameters(results)
    clinical_report = clinical_optimizer.generate_clinical_report(
        results, results.get('validation_results', {})
    )

    print(f"\n📊 임상 최적화 완료!")

    # 권장사항 출력
    frame_opt = optimization.get('frame_rate_optimization', {})
    detection_opt = optimization.get('detection_threshold_optimization', {})

    print(f"⚙️  처리 파라미터 권장사항:")
    print(f"   FPS: {frame_opt.get('recommended_fps', 30)}")
    print(f"   임계값: {detection_opt.get('recommended_visibility_threshold', 0.5)}")

    grade = clinical_report['performance_assessment'].get('overall_grade', 'N/A')
    print(f"🏆 시스템 성능 등급: {grade}")

    # 보고서 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    clinical_optimizer.save_clinical_report(
        clinical_report,
        f"clinical_optimization_report_{timestamp}.json"
    )

def show_help():
    """도움말 표시"""
    print("\n📖 Enhanced MediaPipe Gait Analysis System v2.0 사용법")
    print("=" * 60)
    print("python3 run_gait_analysis.py [옵션]")
    print()
    print("옵션:")
    print("  --quick-test        3명 피험자 빠른 테스트 (약 5분)")
    print("  --full-validation   전체 21명 피험자 검증 (약 30-60분)")
    print("  --single-subject N  특정 피험자 N만 분석")
    print("  --clinical-only     기존 결과로 임상 최적화만 수행")
    print("  --help             이 도움말 표시")
    print()
    print("예제:")
    print("  python3 run_gait_analysis.py --quick-test")
    print("  python3 run_gait_analysis.py --single-subject 1")
    print("  python3 run_gait_analysis.py --full-validation")
    print()
    print("📁 결과 파일:")
    print("  • batch_validation_results_*.json - 배치 분석 결과")
    print("  • *_report_*.json - 임상 보고서")
    print("  • waveform_plots/ - 파형 비교 그래프")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Enhanced MediaPipe Gait Analysis System v2.0')

    parser.add_argument('--quick-test', action='store_true',
                       help='3명 피험자 빠른 테스트')
    parser.add_argument('--full-validation', action='store_true',
                       help='전체 21명 피험자 검증')
    parser.add_argument('--single-subject', type=int, metavar='N',
                       help='특정 피험자 N만 분석')
    parser.add_argument('--clinical-only', action='store_true',
                       help='임상 최적화만 수행')
    parser.add_argument('--help-extended', action='store_true',
                       help='확장 도움말 표시')

    args = parser.parse_args()

    # 배너 출력
    print_banner()

    try:
        if args.help_extended:
            show_help()
        elif args.quick_test:
            run_quick_test()
        elif args.full_validation:
            run_full_validation()
        elif args.single_subject:
            run_single_subject(args.single_subject)
        elif args.clinical_only:
            run_clinical_optimization()
        else:
            # 기본값: 빠른 테스트
            print("ℹ️  옵션이 지정되지 않았습니다. 빠른 테스트를 실행합니다.")
            print("💡 다른 옵션을 보려면: python3 run_gait_analysis.py --help-extended")
            time.sleep(2)
            run_quick_test()

    except KeyboardInterrupt:
        print("\n\n❌ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("💡 --help-extended 옵션으로 사용법을 확인하세요.")

    print(f"\n🏁 Enhanced MediaPipe Gait Analysis System 종료")
    print(f"📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()