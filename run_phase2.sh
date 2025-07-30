#!/bin/bash

echo "=================================="
echo "BdSLW60 SPOTER Research Pipeline"
echo "Phase 2: Dataset Analysis & Preprocessing"
echo "=================================="

# Set up environment
echo "Setting up environment..."
cd src

# Run dataset analysis first
echo "\n1. Running dataset analysis..."
python dataset_analysis.py

# Run quick test of the pipeline
echo "\n2. Running pipeline quick test..."
python main_pipeline.py --quick_test

# If quick test succeeds, offer to run full pipeline
echo "\n3. Quick test completed!"
echo "To run full pipeline: python main_pipeline.py"
echo "To run specific components:"
echo "  - Download dataset: python download_dataset.py"
echo "  - Extract poses: python pose_extraction.py"
echo "  - Normalize poses: python pose_normalization.py"

echo "\nPhase 2 setup complete!"