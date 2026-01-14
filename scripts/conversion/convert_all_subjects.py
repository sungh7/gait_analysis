#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path
import glob

def convert_all_subjects():
    """모든 대상자의 데이터를 표준 형식으로 변환합니다."""

    # 모든 _edited.csv 파일 찾기
    csv_files = glob.glob("data/*/excel/*_edited.csv")
    csv_files.sort()

    print(f"🔄 총 {len(csv_files)}개의 파일을 변환합니다...")

    # 출력 디렉토리 생성
    output_dir = Path("ground_truth_formatted")
    output_dir.mkdir(exist_ok=True)

    successful_conversions = 0
    failed_conversions = 0

    for csv_file in csv_files:
        # 출력 파일명 생성
        subject_id = Path(csv_file).stem.replace('_edited', '')
        output_file = output_dir / f"{subject_id}_ground_truth.xlsx"

        print(f"\n📊 변환 중: {subject_id}")

        try:
            # create_formatted_ground_truth.py 실행
            result = subprocess.run([
                sys.executable, "create_formatted_ground_truth.py",
                "--source", csv_file,
                "--output", str(output_file),
                "--params", "validation_ready_dataset.csv"
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print(f"✅ {subject_id} 변환 완료")
                successful_conversions += 1
            else:
                print(f"❌ {subject_id} 변환 실패: {result.stderr}")
                failed_conversions += 1

        except subprocess.TimeoutExpired:
            print(f"⏰ {subject_id} 변환 시간 초과")
            failed_conversions += 1
        except Exception as e:
            print(f"❌ {subject_id} 변환 오류: {e}")
            failed_conversions += 1

    print(f"\n=== 변환 완료 ===")
    print(f"✅ 성공: {successful_conversions}개")
    print(f"❌ 실패: {failed_conversions}개")
    print(f"📁 출력 디렉토리: {output_dir}")

    if successful_conversions > 0:
        print(f"\n다음 단계: 검증 시스템 실행")
        print(f"  python3 -m core_modules.validation_system --ground_truth_dir {output_dir}")

if __name__ == "__main__":
    convert_all_subjects()