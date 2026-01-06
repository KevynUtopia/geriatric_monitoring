#!/usr/bin/env python3
"""
Verification script to check if the pose extraction system is set up correctly.
"""

import sys
import os
import importlib
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 7):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} is not supported. Please use Python 3.7+")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - OK")
    return True


def check_dependencies():
    """Check if required dependencies are available."""
    print("\nChecking dependencies...")
    required_packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV (opencv-python)',
        'numpy': 'NumPy',
        'ultralytics': 'Ultralytics YOLO',
        'pathlib': 'pathlib (built-in)',
        'json': 'JSON (built-in)',
        'argparse': 'argparse (built-in)'
    }
    
    missing_packages = []
    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install with: pip install torch ultralytics opencv-python numpy")
        return False
    
    return True


def check_files():
    """Check if required files exist."""
    print("\nChecking files...")
    required_files = [
        'pseudo_label_generation.py',
        'video_engine.py',
        'run_pose_extraction.sh',
        'example_usage.py'
    ]
    
    missing_files = []
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} - OK")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    return len(missing_files) == 0


def check_imports():
    """Check if the main modules can be imported."""
    print("\nChecking module imports...")
    
    try:
        import pseudo_label_generation
        print("✅ pseudo_label_generation.py - Can import")
    except Exception as e:
        print(f"❌ pseudo_label_generation.py - Import error: {e}")
        return False
    
    try:
        import video_engine
        print("✅ video_engine.py - Can import")
    except Exception as e:
        print(f"❌ video_engine.py - Import error: {e}")
        return False
    
    return True


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠️  No GPU available, will use CPU (slower)")
    except ImportError:
        print("❌ Cannot check GPU (PyTorch not available)")


def check_script_permissions():
    """Check if bash script is executable."""
    print("\nChecking script permissions...")
    script_path = Path('run_pose_extraction.sh')
    if script_path.exists():
        if os.access(script_path, os.X_OK):
            print("✅ run_pose_extraction.sh is executable")
        else:
            print("⚠️  run_pose_extraction.sh is not executable")
            print("   Run: chmod +x run_pose_extraction.sh")
    else:
        print("❌ run_pose_extraction.sh not found")


def run_syntax_check():
    """Run basic syntax check on main files."""
    print("\nRunning syntax checks...")
    
    files_to_check = ['pseudo_label_generation.py', 'video_engine.py']
    
    for file in files_to_check:
        try:
            with open(file, 'r') as f:
                code = f.read()
            compile(code, file, 'exec')
            print(f"✅ {file} - Syntax OK")
        except SyntaxError as e:
            print(f"❌ {file} - Syntax error: {e}")
            return False
        except FileNotFoundError:
            print(f"❌ {file} - File not found")
            return False
    
    return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Video Pose Extraction System - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Required Files", check_files),
        ("Module Imports", check_imports),
        ("Syntax Check", run_syntax_check),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} - Error: {e}")
            all_passed = False
    
    # Additional checks (non-critical)
    check_gpu()
    check_script_permissions()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All critical checks passed! System is ready to use.")
        print("\nQuick start:")
        print("1. Create sample directory: python3 example_usage.py --create_sample")
        print("2. Add videos to sample_videos/ directory")
        print("3. Run processing: ./run_pose_extraction.sh -i sample_videos -o sample_output")
        print("4. Check output: images/ and labels/ folders with day_camera_video_framenum.jpg/txt")
    else:
        print("❌ Some checks failed. Please fix the issues above before using the system.")
    print("=" * 60)


if __name__ == "__main__":
    main() 