import sys
from pathlib import Path
import argparse
import json

# Import our modules
from download_dataset import download_bdslw60_dataset, analyze_dataset_structure
from dataset_analysis import BdSLW60Analyzer
from pose_extraction import MediaPipePoseExtractor
from pose_normalization import SPOTERPoseNormalizer


class BdSLW60Pipeline:
    """
    Main pipeline for BdSLW60 dataset processing for SPOTER implementation
    """

    def __init__(self, data_dir="data", output_dir="processed_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.analyzer = BdSLW60Analyzer(data_dir)
        self.pose_extractor = MediaPipePoseExtractor()
        self.pose_normalizer = SPOTERPoseNormalizer()  # Keep this name for consistency

    def run_complete_pipeline(self):
        """
        Run the complete preprocessing pipeline
        """
        print("="*60)
        print("BdSLW60 PREPROCESSING PIPELINE FOR SPOTER")
        print("="*60)

        # Step 1: Download and verify dataset
        print("\nStep 1: Dataset Download and Verification")
        print("-" * 40)
        try:
            download_success = download_bdslw60_dataset(str(self.data_dir))
            if not download_success:
                print("Please download the dataset manually and place it in the data directory")
                print("Continue with analysis of existing data...")
        except Exception as e:
            print(f"Download error: {e}")
            print("Continuing with existing data...")

        # Step 2: Dataset analysis
        print("\nStep 2: Dataset Structure Analysis")
        print("-" * 40)
        analysis_results = self.analyzer.analyze_dataset_structure()
        
        # Get video files from analysis or find them manually
        video_files = self._find_video_files()

        if not video_files:
            print("No video files found! Please check dataset download.")
            return False

        # Step 3: Pose extraction
        print(f"\nStep 3: Pose Extraction from {len(video_files)} videos")
        print("-" * 40)

        landmarks_dir = self.output_dir / "landmarks"
        landmarks_dir.mkdir(exist_ok=True)

        # Process videos (limit to first 5 for testing)
        test_videos = video_files[:5]  # Remove this limit for full processing

        processed_landmarks = {}
        for i, video_file in enumerate(test_videos):
            print(f"Processing video {i+1}/{len(test_videos)}: {video_file.name}")

            try:
                # Use the correct method name
                landmarks_data = self.pose_extractor.extract_pose_from_video(video_file)

                if landmarks_data and landmarks_data['pose_landmarks']:
                    video_name = video_file.stem
                    processed_landmarks[video_name] = landmarks_data

                    # Save individual landmarks file
                    landmarks_file = landmarks_dir / f"{video_name}_landmarks.json"
                    self.pose_extractor.save_landmarks(landmarks_data, landmarks_file)

                    print(f"  ✓ Extracted {len(landmarks_data['frames'])} frames")
                else:
                    print(f"  ✗ Failed to extract landmarks")

            except Exception as e:
                print(f"  ✗ Error processing {video_file.name}: {e}")

        print(f"Pose extraction complete! Processed {len(processed_landmarks)} videos")

        # Step 4: Pose normalization
        print("\nStep 4: SPOTER-style Pose Normalization")
        print("-" * 40)

        normalized_dir = self.output_dir / "normalized"
        normalized_dir.mkdir(exist_ok=True)

        normalized_count = 0
        for video_name, landmarks_data in processed_landmarks.items():
            try:
                print(f"Normalizing {video_name}...")

                # Apply SPOTER normalization - pass only pose landmarks
                normalized_sequence = self.pose_normalizer.normalize_pose_sequence(
                    landmarks_data['pose_landmarks']
                )

                # Convert to serializable format
                serializable_data = {
                    'video_name': video_name,
                    'normalized_landmarks': [
                        frame.tolist() if frame is not None else None 
                        for frame in normalized_sequence
                    ],
                    'total_frames': len(normalized_sequence),
                    'original_video_info': landmarks_data['video_info']
                }

                # Save normalized data
                normalized_file = normalized_dir / f"{video_name}_normalized.json"
                with open(normalized_file, 'w') as f:
                    json.dump(serializable_data, f, indent=2)

                normalized_count += 1
                print(f"  ✓ Normalized and saved")

            except Exception as e:
                print(f"  ✗ Error normalizing {video_name}: {e}")

        print(f"Normalization complete! Processed {normalized_count} videos")

        # Step 5: Generate analysis report
        print("\nStep 5: Analysis Report Generation")
        print("-" * 40)

        self.analyzer.compare_with_spoter_dataset()
        self.analyzer.generate_visualization_report(str(self.output_dir))
        self.analyzer.save_analysis_report(str(self.output_dir / "analysis_report.json"))

        # Step 6: Create training data format
        print("\nStep 6: Creating Training Data Format")
        print("-" * 40)

        self._create_training_data_format(normalized_dir)

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)

        print(f"Results saved in: {self.output_dir}")
        print("Next steps:")
        print("1. Review analysis report in results/")
        print("2. Check normalized landmarks in processed_data/normalized/")
        print("3. Use training data format for SPOTER model")

        return True

    def _find_video_files(self):
        """Find all video files in the dataset directory"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(list(self.data_dir.rglob(f'*{ext}')))
        
        return video_files

    def _find_test_video(self):
        """Find a single video file for testing"""
        video_files = self._find_video_files()
        return video_files[0] if video_files else None

    def _create_training_data_format(self, normalized_dir):
        """
        Create training data format compatible with SPOTER model
        """
        training_dir = self.output_dir / "training_format"
        training_dir.mkdir(exist_ok=True)

        # Collect all normalized files
        normalized_files = list(normalized_dir.glob("*_normalized.json"))

        training_data = {
            'videos': [],
            'labels': [],
            'video_info': {},
            'dataset_info': {
                'name': 'BdSLW60',
                'total_videos': len(normalized_files),
                'preprocessing': 'SPOTER_normalization'
            }
        }

        for norm_file in normalized_files:
            with open(norm_file, 'r') as f:
                data = json.load(f)

            video_name = norm_file.stem.replace('_normalized', '')

            # Get the normalized landmarks
            normalized_landmarks = data.get('normalized_landmarks', [])
            
            # Extract feature vectors for each frame
            feature_sequences = []
            for frame_landmarks in normalized_landmarks:
                if frame_landmarks is not None:
                    # Create 108-dimensional feature vector (SPOTER format)
                    pose_landmarks = frame_landmarks
                    feature_vector = self.pose_normalizer.create_combined_feature_vector(pose_landmarks)
                    feature_sequences.append(feature_vector.tolist())

            # Extract label from filename (this depends on dataset structure)
            label = self._extract_label_from_filename(video_name)

            training_data['videos'].append({
                'name': video_name,
                'features': feature_sequences,
                'num_frames': len(feature_sequences)
            })
            training_data['labels'].append(label)
            training_data['video_info'][video_name] = {
                'label': label,
                'num_frames': len(feature_sequences),
                'original_file': norm_file.name
            }

        # Save training data
        training_file = training_dir / "bdslw60_training_data.json"
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2)

        print(f"Training data format saved to {training_file}")
        print(f"Format: {len(training_data['videos'])} videos with 108-dim feature sequences")

        # Create label mapping
        unique_labels = list(set(training_data['labels']))
        label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

        with open(training_dir / "label_mapping.json", 'w') as f:
            json.dump(label_mapping, f, indent=2)

        print(f"Label mapping created: {len(unique_labels)} unique classes")

    def _extract_label_from_filename(self, filename):
        """
        Extract sign class label from filename
        This needs to be adapted based on actual BdSLW60 naming convention
        """
        # For BdSLW60, the label is typically the folder name or part of filename
        # This is a placeholder - adapt based on actual dataset structure
        if '_' in filename:
            return filename.split('_')[0]
        elif '-' in filename:
            return filename.split('-')[0]
        else:
            # Fallback: use first few characters
            return filename[:10]

    def run_quick_test(self):
        """Run quick test of the complete pipeline"""
        print("Running quick test pipeline...")
        
        try:
            # Find a test video
            test_video = self._find_test_video()
            if not test_video:
                print("❌ No test video found")
                return False
            
            print(f"Testing with: {test_video.name}")
            
            # Extract poses - using the correct method name
            landmarks_data = self.pose_extractor.extract_pose_from_video(test_video)
            
            if not landmarks_data['pose_landmarks']:
                print("❌ No pose landmarks extracted")
                return False
            
            # Normalize poses - this returns a list, not a dict
            normalized_sequence = self.pose_normalizer.normalize_pose_sequence(
                landmarks_data['pose_landmarks']
            )
            
            print("✓ Normalized data")
            
            # Check results - Fix: normalized_sequence is a list, not a dict
            if len(normalized_sequence) > 0:
                print(f"✅ Quick test successful!")
                print(f"   • Processed {len(landmarks_data['pose_landmarks'])} frames")
                print(f"   • Successfully normalized {len(normalized_sequence)} frames")
                
                # Save test results
                test_output_dir = Path("processed_data/test_results")
                test_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save sample normalized data
                test_data = {
                    'video_name': test_video.name,
                    'total_frames': len(landmarks_data['pose_landmarks']),
                    'normalized_frames': len(normalized_sequence),
                    'sample_normalized_frame': normalized_sequence[0].tolist() if len(normalized_sequence) > 0 and normalized_sequence[0] is not None else None
                }
                
                with open(test_output_dir / "quick_test_results.json", 'w') as f:
                    json.dump(test_data, f, indent=2)
                
                return True
            else:
                print("❌ Normalization failed - no frames were normalized")
                return False
                
        except Exception as e:
            print(f"❌ Quick test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_dataset_analysis_only(self):
        """Run only dataset analysis"""
        print("Running dataset analysis only...")
        
        try:
            # Dataset analysis
            print("🔍 Analyzing dataset structure...")
            analysis_results = self.analyzer.analyze_dataset_structure()
            
            print("📊 Comparing with SPOTER dataset...")
            comparison = self.analyzer.compare_with_spoter_dataset()
            
            print("📈 Generating visualization report...")
            viz_path = self.analyzer.generate_visualization_report(str(self.output_dir))
            
            print("💾 Saving analysis report...")
            report_path = self.analyzer.save_analysis_report(str(self.output_dir))
            
            print("✅ Dataset analysis completed successfully!")
            print(f"   • Analysis report: {report_path}")
            print(f"   • Visualizations: {viz_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Dataset analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='BdSLW60 Preprocessing Pipeline for SPOTER')
    parser.add_argument('--data_dir', default='data', help='Dataset directory')
    parser.add_argument('--output_dir', default='processed_data', help='Output directory')
    parser.add_argument('--quick_test', action='store_true', help='Run quick test instead of full pipeline')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = BdSLW60Pipeline(args.data_dir, args.output_dir)

    # Run pipeline
    if args.quick_test:
        success = pipeline.run_quick_test()
    else:
        success = pipeline.run_complete_pipeline()

    if success:
        print("\nPipeline execution completed!")
    else:
        print("\nPipeline execution failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
