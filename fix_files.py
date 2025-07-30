import shutil
import os

print("Fixing BdSLW60 SPOTER Research Pipeline Files")
print("==============================================")

# Backup original files
print("Creating backup of original files...")
try:
    shutil.copy('src/dataset_analysis.py', 'src/dataset_analysis_backup.py')
    shutil.copy('src/pose_normalization.py', 'src/pose_normalization_backup.py')
    print("✓ Backups created")
except Exception as e:
    print(f"Backup warning: {e}")

# Replace with corrected files
print("Replacing with corrected files...")
try:
    shutil.copy('dataset_analysis_corrected.py', 'src/dataset_analysis.py')
    shutil.copy('pose_normalization_corrected.py', 'src/pose_normalization.py')
    print("✓ Files replaced successfully!")
    print("You can now run: python run_phase2.py")
except Exception as e:
    print(f"Error replacing files: {e}")