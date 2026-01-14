"""
Batch conversion script for gait analysis Excel files.
Converts all Excel files in /data/gait/data/*/excel/*.xlsx to structured formats.
"""

import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from gait_parser import GaitDataParser
import warnings


def find_all_excel_files(base_dir: str = "/data/gait/data") -> list:
    """
    Find all Excel files in the data directory.

    Args:
        base_dir: Base directory to search

    Returns:
        List of tuples (subject_id, file_path)
    """
    files = []
    base_path = Path(base_dir)

    for excel_file in base_path.glob("*/excel/*.xlsx"):
        # Extract subject ID from path (e.g., data/1/excel/S1_01.xlsx -> S1_01)
        subject_id = excel_file.stem
        files.append((subject_id, str(excel_file)))

    return sorted(files)


def convert_single_file(subject_id: str, file_path: str, output_dir: str) -> dict:
    """
    Convert a single Excel file to structured formats.

    Args:
        subject_id: Subject identifier
        file_path: Path to Excel file
        output_dir: Output directory

    Returns:
        Dictionary with conversion status and details
    """
    result = {
        'subject_id': subject_id,
        'file_path': file_path,
        'success': False,
        'error': None,
        'outputs': {}
    }

    try:
        # Parse data
        parser = GaitDataParser(file_path)

        # Validate data
        is_valid, message = parser.validate_data()
        if not is_valid:
            result['error'] = f"Validation failed: {message}"
            return result

        # Extract subject info
        subject_info = parser.extract_subject_info()
        info_file = os.path.join(output_dir, f"{subject_id}_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(subject_info, f, indent=2, ensure_ascii=False, default=str)
        result['outputs']['info'] = info_file

        # Extract gait data in long format
        gait_long = parser.extract_gait_data_long(subject_id=subject_id)
        long_file = os.path.join(output_dir, f"{subject_id}_gait_long.csv")
        gait_long.to_csv(long_file, index=False)
        result['outputs']['gait_long'] = long_file

        result['success'] = True
        result['record_count'] = len(gait_long)

    except Exception as e:
        result['error'] = str(e)

    return result


def combine_all_subjects(output_dir: str, subject_results: list) -> str:
    """
    Combine all subjects' gait data into a single CSV file.

    Args:
        output_dir: Output directory
        subject_results: List of conversion results

    Returns:
        Path to combined CSV file
    """
    all_data = []

    for result in subject_results:
        if result['success'] and 'gait_long' in result['outputs']:
            df = pd.read_csv(result['outputs']['gait_long'])
            all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_file = os.path.join(output_dir, "all_subjects_combined.csv")
        combined_df.to_csv(combined_file, index=False)
        return combined_file

    return None


def create_summary_report(output_dir: str, results: list) -> str:
    """
    Create a summary report of the conversion process.

    Args:
        output_dir: Output directory
        results: List of conversion results

    Returns:
        Path to summary report
    """
    summary = {
        'total_files': len(results),
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'total_records': sum(r.get('record_count', 0) for r in results if r['success']),
        'subjects': []
    }

    for result in results:
        summary['subjects'].append({
            'subject_id': result['subject_id'],
            'success': result['success'],
            'error': result['error'],
            'record_count': result.get('record_count', 0)
        })

    summary_file = os.path.join(output_dir, "conversion_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary_file


def main():
    """Main conversion process."""
    # Configuration
    BASE_DIR = "/data/gait/data"
    OUTPUT_DIR = "/data/gait/processed"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Gait Analysis Data Conversion")
    print("=" * 60)

    # Find all Excel files
    print(f"\nSearching for Excel files in {BASE_DIR}...")
    files = find_all_excel_files(BASE_DIR)
    print(f"Found {len(files)} files to process\n")

    if not files:
        print("No Excel files found. Exiting.")
        return

    # Convert each file
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for subject_id, file_path in tqdm(files, desc="Converting files"):
            result = convert_single_file(subject_id, file_path, OUTPUT_DIR)
            results.append(result)

            if not result['success']:
                print(f"\n  ⚠️  Failed: {subject_id} - {result['error']}")

    # Combine all subjects
    print("\nCombining all subjects' data...")
    combined_file = combine_all_subjects(OUTPUT_DIR, results)

    # Create summary report
    summary_file = create_summary_report(OUTPUT_DIR, results)

    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    total_records = sum(r.get('record_count', 0) for r in results if r['success'])

    print(f"Total files processed: {len(results)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"Total records: {total_records:,}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  - Individual files: {successful} × 2 (info.json + gait_long.csv)")
    if combined_file:
        print(f"  - Combined file: {combined_file}")
    print(f"  - Summary report: {summary_file}")

    if failed > 0:
        print(f"\n⚠️  {failed} file(s) failed. Check {summary_file} for details.")

    print("=" * 60)


if __name__ == "__main__":
    main()
