import os
import zipfile
from pathlib import Path
import requests

def download_with_manual_method():
    """
    Alternative download method if Kaggle API fails
    """
    print("Manual Download Instructions:")
    print("=" * 40)
    print("1. Go to: https://www.kaggle.com/datasets/hasaniut/bdslw60")
    print("2. Click 'Download' button")
    print("3. Extract the zip file to the 'data' folder")
    print("4. Run this script again to analyze the dataset")

def setup_kaggle_credentials():
    """
    Guide user to setup Kaggle credentials
    """
    print("Kaggle API Setup Instructions:")
    print("=" * 35)
    print("1. Go to kaggle.com and sign in")
    print("2. Go to Account settings")
    print("3. Scroll to API section")
    print("4. Click 'Create New API Token'")
    print("5. Download kaggle.json file")
    print("6. Place it in ~/.kaggle/ directory")
    print("7. Set permissions: chmod 600 ~/.kaggle/kaggle.json")

def download_bdslw60_dataset(data_dir="data"):
    """
    Download BdSLW60 dataset from Kaggle
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    print(f"Attempting to download BdSLW60 dataset to {data_dir}...")

    try:
        # Try to import and use Kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        # Dataset information
        dataset_name = "hasaniut/bdslw60"

        print(f"Downloading {dataset_name}...")

        # Download dataset
        api.dataset_download_files(
            dataset_name, 
            path=data_dir, 
            unzip=True
        )
        print(f"Dataset downloaded successfully to {data_dir}")
        return True

    except ImportError:
        print("Kaggle library not installed. Please install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print()
        setup_kaggle_credentials()
        print()
        download_with_manual_method()
        return False

def analyze_dataset_structure(data_dir="data"):
    """
    Analyze the structure of downloaded dataset
    """
    print("\n" + "="*50)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*50)

    dataset_path = Path(data_dir)

    if not dataset_path.exists():
        print(f"Dataset directory {data_dir} not found!")
        return

    # Count files by extension
    file_counts = {}
    total_size = 0
    video_files = []

    for file in dataset_path.rglob("*"):
        if file.is_file():
            ext = file.suffix.lower()
            file_counts[ext] = file_counts.get(ext, 0) + 1
            total_size += file.stat().st_size

            # Collect video files
            if ext in ['.mp4', '.avi', '.mov', '.mkv']:
                video_files.append(file)

    print(f"Total files: {sum(file_counts.values())}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print("\nFile types:")
    for ext, count in sorted(file_counts.items()):
        print(f"  {ext or 'no extension'}: {count} files")

    # Look for main dataset folders/files
    main_items = [item for item in dataset_path.iterdir()]
    print("\nMain dataset items:")
    for item in main_items:
        if item.is_dir():
            file_count = len(list(item.rglob("*")))
            print(f"  📁 {item.name}/ ({file_count} items)")
        else:
            size_mb = item.stat().st_size / (1024**2)
            print(f"  📄 {item.name} ({size_mb:.2f} MB)")

    # Analyze video files if found
    if video_files:
        print(f"\nFound {len(video_files)} video files")
        print("Sample video files:")
        for i, video in enumerate(video_files[:5]):  # Show first 5
            size_mb = video.stat().st_size / (1024**2)
            print(f"  {video.name} ({size_mb:.2f} MB)")

        if len(video_files) > 5:
            print(f"  ... and {len(video_files) - 5} more videos")

if __name__ == "__main__":
    print("BdSLW60 Dataset Downloader")
    print("=" * 30)

    # Try to download dataset
    success = download_bdslw60_dataset()

    # Always try to analyze what we have
    analyze_dataset_structure()