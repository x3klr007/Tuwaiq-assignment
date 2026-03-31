#!/usr/bin/env python3
"""
Release Check Script for My-Capstone Project

Performs Red Team audit validation before packaging:
1. File verification
2. ASCII character scanning
3. Portability check (relative paths)
4. Dependency audit
5. JSON integrity validation

If all checks pass, creates submission package.

Author: Release Engineering
Version: 1.0.0
"""

import os
import sys
import json
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

REQUIRED_FILES = {
    'notebooks': [
        '01_cleaning.ipynb',
        '02_features.ipynb',
        '03_eda.ipynb',
        '04_modeling.ipynb',
        '05_fairness.ipynb'
    ],
    'src': ['utils.py'],
    'root': ['requirements.txt', 'README.md'],
    'reports': ['FINAL_REPORT.md']
}

SUBMISSION_NAME = 'submission_HAMAD_ALDHUBAYB'
ZIP_NAME = 'submission.zip'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(text: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def print_result(check_name: str, passed: bool, details: str = "") -> None:
    """Print check result with status."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {check_name}")
    if details:
        print(f"       {details}")


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    return os.path.isfile(filepath)


def find_non_ascii_bytes(filepath: str) -> List[Tuple[int, int]]:
    """
    Find all non-ASCII bytes in a file.
    
    Returns:
        List of (position, byte_value) tuples
    """
    non_ascii = []
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
        for i, byte in enumerate(content):
            if byte > 127:
                non_ascii.append((i, byte))
    except Exception as e:
        print(f"       Error reading {filepath}: {e}")
    return non_ascii


def check_absolute_paths(filepath: str) -> List[str]:
    """
    Scan file for absolute path patterns.
    
    Returns:
        List of found absolute path strings
    """
    absolute_patterns = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            # Check for Unix absolute paths
            if '/home/' in line or '/root/' in line or '/var/' in line:
                if not line.strip().startswith('#'):
                    absolute_patterns.append(f"Line {line_num}: {line.strip()[:60]}")
            # Check for Windows absolute paths
            if 'C:\\' in line or 'D:\\' in line:
                if not line.strip().startswith('#'):
                    absolute_patterns.append(f"Line {line_num}: {line.strip()[:60]}")
    except Exception as e:
        print(f"       Error scanning {filepath}: {e}")
    return absolute_patterns


def validate_json_file(filepath: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a file is valid JSON.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            json.load(f)
        return True, None
    except json.JSONDecodeError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def audit_requirements(filepath: str) -> Tuple[bool, List[str]]:
    """
    Check requirements.txt for correct dependency naming.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                # Check for wrong package name
                if line.startswith('sklearn'):
                    issues.append(f"Line {line_num}: Uses 'sklearn' instead of 'scikit-learn'")
        
        # Check for required packages
        required = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']
        for pkg in required:
            if pkg not in content:
                issues.append(f"Missing required package: {pkg}")
        
        return len(issues) == 0, issues
    except Exception as e:
        return False, [str(e)]


# =============================================================================
# CHECK FUNCTIONS
# =============================================================================

def check_file_verification() -> Tuple[bool, List[str]]:
    """
    Check 1: Verify all required files exist.
    
    Returns:
        Tuple of (all_exist, list_of_missing_files)
    """
    print_header("CHECK 1: FILE VERIFICATION")
    
    missing_files = []
    all_exist = True
    
    # Check notebooks
    for nb in REQUIRED_FILES['notebooks']:
        path = f"notebooks/{nb}"
        exists = check_file_exists(path)
        print_result(f"notebooks/{nb}", exists)
        if not exists:
            missing_files.append(path)
            all_exist = False
    
    # Check src files
    for src_file in REQUIRED_FILES['src']:
        path = f"src/{src_file}"
        exists = check_file_exists(path)
        print_result(f"src/{src_file}", exists)
        if not exists:
            missing_files.append(path)
            all_exist = False
    
    # Check root files
    for root_file in REQUIRED_FILES['root']:
        exists = check_file_exists(root_file)
        print_result(f"{root_file}", exists)
        if not exists:
            missing_files.append(root_file)
            all_exist = False
    
    # Check reports
    for report in REQUIRED_FILES['reports']:
        path = f"reports/{report}"
        exists = check_file_exists(path)
        print_result(f"reports/{report}", exists)
        if not exists:
            missing_files.append(path)
            all_exist = False
    
    return all_exist, missing_files


def check_ascii_compliance() -> Tuple[bool, List[str]]:
    """
    Check 2: Scan all Python, notebook, and markdown files for non-ASCII.
    
    Returns:
        Tuple of (all_clean, list_of_files_with_issues)
    """
    print_header("CHECK 2: ASCII COMPLIANCE SCAN")
    
    files_to_check = []
    issues = []
    all_clean = True
    
    # Collect all target files
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and submission folders
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != SUBMISSION_NAME]
        
        for file in files:
            if file.endswith(('.py', '.ipynb', '.md', '.txt')):
                filepath = os.path.join(root, file)
                # Skip the release_check.py script itself
                if 'release_check.py' not in filepath:
                    files_to_check.append(filepath)
    
    # Check each file
    for filepath in sorted(files_to_check):
        non_ascii = find_non_ascii_bytes(filepath)
        if non_ascii:
            all_clean = False
            issues.append(filepath)
            # Show first 3 violations
            details = "; ".join([f"pos {pos}: 0x{byte:02X}" for pos, byte in non_ascii[:3]])
            if len(non_ascii) > 3:
                details += f" (+{len(non_ascii) - 3} more)"
            print_result(filepath, False, details)
        else:
            print_result(filepath, True)
    
    return all_clean, issues


def check_portability() -> Tuple[bool, List[str]]:
    """
    Check 3: Scan for absolute paths that would break on other systems.
    
    Returns:
        Tuple of (all_relative, list_of_files_with_absolute_paths)
    """
    print_header("CHECK 3: PATH PORTABILITY")
    
    files_to_check = []
    issues = []
    all_relative = True
    
    # Collect Python and notebook files
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != SUBMISSION_NAME]
        for file in files:
            if file.endswith(('.py', '.ipynb')):
                filepath = os.path.join(root, file)
                if 'release_check.py' not in filepath:
                    files_to_check.append(filepath)
    
    # Check each file
    for filepath in sorted(files_to_check):
        absolute_paths = check_absolute_paths(filepath)
        if absolute_paths:
            all_relative = False
            issues.append(filepath)
            print_result(filepath, False)
            for detail in absolute_paths[:2]:
                print(f"       {detail}")
        else:
            print_result(filepath, True)
    
    return all_relative, issues


def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check 4: Verify requirements.txt has correct dependency naming.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    print_header("CHECK 4: DEPENDENCY AUDIT")
    
    if not check_file_exists('requirements.txt'):
        print_result("requirements.txt exists", False, "File not found")
        return False, ["requirements.txt not found"]
    
    is_valid, issues = audit_requirements('requirements.txt')
    
    if is_valid:
        print_result("Dependency naming", True, "Uses 'scikit-learn' (correct)")
        print_result("Required packages", True, "All required packages present")
    else:
        for issue in issues:
            print_result("Dependency check", False, issue)
    
    return is_valid, issues


def check_json_integrity() -> Tuple[bool, List[str]]:
    """
    Check 5: Validate all notebooks are valid JSON.
    
    Returns:
        Tuple of (all_valid, list_of_corrupted_files)
    """
    print_header("CHECK 5: JSON INTEGRITY")
    
    corrupted = []
    all_valid = True
    
    for nb in REQUIRED_FILES['notebooks']:
        path = f"notebooks/{nb}"
        if check_file_exists(path):
            is_valid, error = validate_json_file(path)
            if is_valid:
                print_result(f"notebooks/{nb}", True)
            else:
                all_valid = False
                corrupted.append(path)
                error_short = error.split('\n')[0][:50]
                print_result(f"notebooks/{nb}", False, error_short)
        else:
            print_result(f"notebooks/{nb}", False, "File not found")
            all_valid = False
    
    return all_valid, corrupted


# =============================================================================
# PACKAGING FUNCTIONS
# =============================================================================

def create_submission_package() -> bool:
    """
    Create the submission package if all checks pass.
    
    Returns:
        True if packaging successful, False otherwise
    """
    print_header("PACKAGING: CREATING SUBMISSION")
    
    try:
        # Remove existing submission directory if present
        if os.path.exists(SUBMISSION_NAME):
            print(f"  Removing existing {SUBMISSION_NAME}/")
            shutil.rmtree(SUBMISSION_NAME)
        
        # Create submission directory structure
        print(f"  Creating {SUBMISSION_NAME}/")
        os.makedirs(SUBMISSION_NAME)
        os.makedirs(f"{SUBMISSION_NAME}/data/raw")
        os.makedirs(f"{SUBMISSION_NAME}/data/processed")
        os.makedirs(f"{SUBMISSION_NAME}/notebooks")
        os.makedirs(f"{SUBMISSION_NAME}/reports")
        os.makedirs(f"{SUBMISSION_NAME}/src")
        
        # Copy files
        print("  Copying files...")
        
        # Notebooks
        for nb in REQUIRED_FILES['notebooks']:
            src = f"notebooks/{nb}"
            dst = f"{SUBMISSION_NAME}/notebooks/{nb}"
            if check_file_exists(src):
                shutil.copy2(src, dst)
                print(f"    Copied: {src}")
        
        # Source files
        for src_file in REQUIRED_FILES['src']:
            src = f"src/{src_file}"
            dst = f"{SUBMISSION_NAME}/src/{src_file}"
            if check_file_exists(src):
                shutil.copy2(src, dst)
                print(f"    Copied: {src}")
        
        # Root files
        for root_file in REQUIRED_FILES['root']:
            if check_file_exists(root_file):
                shutil.copy2(root_file, f"{SUBMISSION_NAME}/{root_file}")
                print(f"    Copied: {root_file}")
        
        # Reports
        for report in REQUIRED_FILES['reports']:
            src = f"reports/{report}"
            dst = f"{SUBMISSION_NAME}/reports/{report}"
            if check_file_exists(src):
                shutil.copy2(src, dst)
                print(f"    Copied: {src}")
        
        # Copy raw data files if they exist
        if os.path.exists('data/raw/insurance.csv'):
            shutil.copy2('data/raw/insurance.csv', f"{SUBMISSION_NAME}/data/raw/")
            print("    Copied: data/raw/insurance.csv")
        
        # Copy processed data files if they exist
        processed_files = [
            'data/processed/insurance_cleaned.csv',
            'data/processed/insurance_engineered.csv',
            'data/processed/insurance_numeric.csv'
        ]
        for proc_file in processed_files:
            if os.path.exists(proc_file):
                shutil.copy2(proc_file, f"{SUBMISSION_NAME}/data/processed/")
                print(f"    Copied: {proc_file}")
        
        # Copy generated report images if they exist
        report_images = [
            'reports/phase1_outlier_comparison.png',
            'reports/phase2_correlation_matrix.png',
            'reports/chart1_charges_by_smoker.png',
            'reports/chart2_age_charges_scatter.png',
            'reports/chart3_bmi_charges_region.png',
            'reports/chart4_charges_heatmap.png',
            'reports/chart5_feature_correlations.png',
            'reports/chart6_age_distribution.png',
            'reports/chart6c_distributions_kde.png',
            'reports/dashboard_2x2_summary.png',
            'reports/residual_analysis_best_model.png',
            'reports/feature_importance_comparison.png',
            'reports/smoker_bias_analysis.png'
        ]
        for img_file in report_images:
            if os.path.exists(img_file):
                shutil.copy2(img_file, f"{SUBMISSION_NAME}/reports/")
                print(f"    Copied: {img_file}")
        
        # Create zip file
        print(f"\n  Creating {ZIP_NAME}...")
        with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(SUBMISSION_NAME):
                for file in files:
                    filepath = os.path.join(root, file)
                    arcname = filepath  # Keep the submission_ prefix in zip
                    zipf.write(filepath, arcname)
                    print(f"    Added: {filepath}")
        
        print(f"\n  [SUCCESS] Submission package created: {ZIP_NAME}")
        print(f"  Contents:")
        print(f"    - {SUBMISSION_NAME}/ directory with all files")
        print(f"    - {ZIP_NAME} compressed archive")
        
        return True
        
    except Exception as e:
        print(f"\n  [ERROR] Packaging failed: {e}")
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> int:
    """
    Main entry point for release check script.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("=" * 70)
    print("RELEASE CHECK - MY-CAPSTONE PROJECT")
    print("Red Team Audit and Packaging Script")
    print("=" * 70)
    
    # Track all check results
    results = {
        'file_verification': False,
        'ascii_compliance': False,
        'path_portability': False,
        'dependency_audit': False,
        'json_integrity': False
    }
    
    # Run all checks
    results['file_verification'], _ = check_file_verification()
    results['ascii_compliance'], _ = check_ascii_compliance()
    results['path_portability'], _ = check_portability()
    results['dependency_audit'], _ = check_dependencies()
    results['json_integrity'], _ = check_json_integrity()
    
    # Final summary
    print_header("FINAL AUDIT SUMMARY")
    
    all_passed = all(results.values())
    
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name.replace('_', ' ').title()}")
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("RESULT: ALL CHECKS PASSED")
        print("=" * 70)
        
        # Create submission package
        success = create_submission_package()
        
        if success:
            print("\n" + "=" * 70)
            print("SUBMISSION READY FOR DELIVERY")
            print("=" * 70)
            print(f"\nPackage: {ZIP_NAME}")
            print(f"Size: {os.path.getsize(ZIP_NAME) / 1024:.1f} KB")
            return 0
        else:
            print("\n[ERROR] Packaging failed but all checks passed.")
            return 1
    else:
        print("RESULT: AUDIT FAILED - SUBMISSION BLOCKED")
        print("=" * 70)
        print("\n[ERROR] One or more critical checks failed.")
        print("Fix all issues before submitting.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
