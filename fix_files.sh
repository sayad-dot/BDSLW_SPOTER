#!/bin/bash

echo "Fixing BdSLW60 SPOTER Research Pipeline Files"
echo "============================================="

# Backup original files
echo "Creating backup of original files..."
cp src/dataset_analysis.py src/dataset_analysis_backup.py
cp src/pose_normalization.py src/pose_normalization_backup.py

# Replace with corrected files
echo "Replacing with corrected files..."
cp dataset_analysis_corrected.py src/dataset_analysis.py
cp pose_normalization_corrected.py src/pose_normalization.py

echo "✓ Files replaced successfully!"
echo "You can now run: python run_phase2.py"