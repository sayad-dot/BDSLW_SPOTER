import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def main():
    print("="*50)
    print("BdSLW60 SPOTER Research Pipeline")
    print("Phase 2: Dataset Analysis & Preprocessing")  
    print("="*50)

    # Check if we're in the right directory
    if not Path('src').exists():
        print("Error: Please run this script from the project root directory")
        print("Expected structure:")
        print("  bdsl_spoter_research/")
        print("    ├── src/")
        print("    ├── data/")
        print("    └── run_phase2.py")
        return

    # Import our modules
    try:
        from src.main_pipeline import BdSLW60Pipeline
        from src.dataset_analysis import BdSLW60Analyzer
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required packages are installed:")
        print("pip install -r requirements.txt")
        return

    # Initialize pipeline
    pipeline = BdSLW60Pipeline("data", "processed_data")

    print("\nChoose an option:")
    print("1. Quick test (recommended first)")
    print("2. Full pipeline")
    print("3. Dataset analysis only")

    choice = input("Enter choice (1-3): ").strip()

    if choice == '1':
        print("\nRunning quick test...")
        success = pipeline.run_quick_test()
    elif choice == '2':
        print("\nRunning full pipeline...")
        success = pipeline.run_complete_pipeline()
    elif choice == '3':
        print("\nRunning dataset analysis...")
        analyzer = BdSLW60Analyzer("data")
        analyzer.analyze_dataset_structure()
        analyzer.compare_with_spoter_dataset()
        success = True
    else:
        print("Invalid choice")
        return

    if success:
        print("\n✓ Phase 2 completed successfully!")
        print("\nNext steps:")
        print("- Review results in processed_data/ directory")
        print("- Check analysis visualizations")
        print("- Proceed to Phase 3: SPOTER Model Implementation")
    else:
        print("\n✗ Phase 2 encountered errors")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()