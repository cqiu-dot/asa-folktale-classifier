#!/usr/bin/env python
"""
Setup validation script: Verify all dependencies and configuration are correct
"""

import sys
import subprocess
from pathlib import Path


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        __import__(import_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError:
        print(f"  ✗ {package_name} (NOT installed)")
        return False


def check_required_packages():
    """Verify all required packages are installed"""
    print("\n" + "="*60)
    print("CHECKING REQUIRED PACKAGES")
    print("="*60)
    
    required = {
        'torch': 'torch',
        'transformers': 'transformers',
        'sentence-transformers': 'sentence_transformers',
        'scikit-learn': 'sklearn',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
    }
    
    missing = []
    for pkg, import_name in required.items():
        if not check_package(pkg, import_name):
            missing.append(pkg)
    
    if missing:
        print(f"\n✗ Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages installed")
        return True


def check_directory_structure():
    """Verify project directory structure"""
    print("\n" + "="*60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*60)
    
    required_dirs = [
        'data',
        'data/western',
        'data/asian',
        'data/processed',
        'src',
        'notebooks',
        'results',
        'results/models',
        'results/embeddings',
        'results/plots',
        'results/analysis',
        'config',
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (NOT found)")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n  Creating missing directories...")
        for dir_path in missing_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {len(missing_dirs)} directories")
        return True
    else:
        print("\n✓ All required directories exist")
        return True


def check_configuration():
    """Verify configuration files"""
    print("\n" + "="*60)
    print("CHECKING CONFIGURATION")
    print("="*60)
    
    config_file = Path('config/config.yaml')
    if config_file.exists():
        print(f"  ✓ config/config.yaml")
        try:
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
            print(f"  ✓ Configuration is valid YAML")
            return True
        except Exception as e:
            print(f"  ✗ Configuration error: {e}")
            return False
    else:
        print(f"  ✗ config/config.yaml (NOT found)")
        return False


def check_data_files():
    """Check if data files are available"""
    print("\n" + "="*60)
    print("CHECKING DATA FILES")
    print("="*60)
    
    western_csv = Path('data/western/aft.csv')
    asian_dir = Path('data/asian')
    
    if western_csv.exists():
        print(f"  ✓ data/western/aft.csv ({western_csv.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f"  ℹ data/western/aft.csv (NOT found - download from GitHub)")
    
    asian_files = list(asian_dir.glob('**/*.txt'))
    if asian_files:
        print(f"  ✓ Found {len(asian_files)} Asian folktale files")
    else:
        print(f"  ℹ data/asian/ (empty - collect from Project Gutenberg)")
    
    return True


def check_python_version():
    """Verify Python version"""
    print("\n" + "="*60)
    print("CHECKING PYTHON VERSION")
    print("="*60)
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 9:
        print(f"  ✓ Python {version_str}")
        return True
    else:
        print(f"  ✗ Python {version_str} (need 3.9+)")
        return False


def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("FOLKTALE CLASSIFIER - SETUP VALIDATION")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'Packages': check_required_packages(),
        'Directory Structure': check_directory_structure(),
        'Configuration': check_configuration(),
        'Data Files': check_data_files(),
    }
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All checks passed! Ready to run notebooks.")
        print("\nNext steps:")
        print("  1. Download data files (Western + Asian folktales)")
        print("  2. Run: jupyter notebook notebooks/01_exploratory_analysis.ipynb")
        sys.exit(0)
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nFor help, see QUICKSTART.md")
        sys.exit(1)


if __name__ == '__main__':
    main()
