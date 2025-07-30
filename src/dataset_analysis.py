import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import seaborn as sns
from pathlib import Path

class BdSLW60Analyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.archive_path = self.dataset_path / "archive"
        self.analysis_results = {}
        
    def analyze_dataset_structure(self):
        """Analyze the complete BdSLW60 dataset structure"""
        print("🔍 Analyzing BdSLW60 dataset structure...")
        
        analysis = {
            'total_classes': 0,
            'total_videos': 0,
            'class_distribution': {},
            'video_info': {},
            'signers_per_class': {},
            'file_formats': Counter(),
            'video_durations': [],
            'resolution_info': {}
        }
        
        if not self.archive_path.exists():
            print(f"❌ Dataset path not found: {self.archive_path}")
            return analysis
            
        # Get all class folders
        class_folders = [f for f in self.archive_path.iterdir() if f.is_dir()]
        analysis['total_classes'] = len(class_folders)
        
        print(f"📊 Found {analysis['total_classes']} classes")
        
        for class_folder in class_folders:
            class_name = class_folder.name
            videos = list(class_folder.glob("*.mp4")) + list(class_folder.glob("*.avi")) + list(class_folder.glob("*.mov"))
            
            analysis['class_distribution'][class_name] = len(videos)
            analysis['total_videos'] += len(videos)
            
            # Analyze video properties for first few videos
            for i, video_path in enumerate(videos[:3]):  # Sample first 3 videos per class
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    analysis['video_info'][f"{class_name}_{i}"] = {
                        'fps': fps,
                        'duration': duration,
                        'frames': frame_count,
                        'resolution': (width, height)
                    }
                    
                    analysis['video_durations'].append(duration)
                    analysis['file_formats'][video_path.suffix.lower()] += 1
                    
                    resolution_key = f"{width}x{height}"
                    if resolution_key not in analysis['resolution_info']:
                        analysis['resolution_info'][resolution_key] = 0
                    analysis['resolution_info'][resolution_key] += 1
                    
                    cap.release()
                except Exception as e:
                    print(f"⚠️  Error analyzing {video_path}: {e}")
        
        self.analysis_results = analysis
        print(f"✅ Analysis complete: {analysis['total_videos']} videos across {analysis['total_classes']} classes")
        return analysis
    
    def compare_with_spoter_dataset(self):
        """Compare BdSL characteristics with SPOTER's original ASL dataset"""
        print("🔄 Comparing BdSL with ASL characteristics...")
        
        # SPOTER paper characteristics (WLASL dataset)
        spoter_characteristics = {
            'num_classes': 2000,
            'avg_video_length': 1.5,  # seconds
            'typical_fps': 30,
            'signing_space': {
                'width_ratio': 0.8,  # proportion of frame width used for signing
                'height_ratio': 0.9,  # proportion of frame height used for signing
                'hand_movement_range': 'full_upper_body'
            }
        }
        
        # Calculate BdSL characteristics
        bdsl_characteristics = {
            'num_classes': self.analysis_results.get('total_classes', 0),
            'avg_video_length': np.mean(self.analysis_results.get('video_durations', [0])),
            'total_videos': self.analysis_results.get('total_videos', 0),
            'videos_per_class': self.analysis_results.get('total_videos', 0) / max(1, self.analysis_results.get('total_classes', 1))
        }
        
        comparison = {
            'bdsl_vs_asl': {
                'classes': f"BdSL: {bdsl_characteristics['num_classes']} vs ASL: {spoter_characteristics['num_classes']}",
                'avg_length': f"BdSL: {bdsl_characteristics['avg_video_length']:.2f}s vs ASL: {spoter_characteristics['avg_video_length']}s",
                'videos_per_class': f"BdSL: {bdsl_characteristics['videos_per_class']:.1f} videos/class"
            },
            'normalization_requirements': {
                'temporal': 'BdSL videos may need different temporal normalization',
                'spatial': 'BdSL signing space might differ from ASL proportions',
                'cultural_gestures': 'BdSL may have unique gesture patterns requiring adaptation'
            }
        }
        
        print("📋 Comparison Results:")
        for key, value in comparison['bdsl_vs_asl'].items():
            print(f"   • {key}: {value}")
            
        return comparison
    
    def generate_visualization_report(self, output_dir="processed_data"):
        """Generate visualization plots for the analysis"""
        print("📊 Generating visualization report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BdSLW60 Dataset Analysis Report', fontsize=16, fontweight='bold')
        
        # Plot 1: Class distribution
        if self.analysis_results.get('class_distribution'):
            classes = list(self.analysis_results['class_distribution'].keys())[:20]  # Top 20 classes
            counts = [self.analysis_results['class_distribution'][c] for c in classes]
            
            axes[0, 0].bar(range(len(classes)), counts)
            axes[0, 0].set_title('Videos per Class (Top 20)')
            axes[0, 0].set_xlabel('Classes')
            axes[0, 0].set_ylabel('Number of Videos')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Video duration distribution
        if self.analysis_results.get('video_durations'):
            durations = [d for d in self.analysis_results['video_durations'] if d > 0]
            axes[0, 1].hist(durations, bins=20, alpha=0.7)
            axes[0, 1].set_title('Video Duration Distribution')
            axes[0, 1].set_xlabel('Duration (seconds)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: File format distribution
        if self.analysis_results.get('file_formats'):
            formats = list(self.analysis_results['file_formats'].keys())
            counts = list(self.analysis_results['file_formats'].values())
            axes[1, 0].pie(counts, labels=formats, autopct='%1.1f%%')
            axes[1, 0].set_title('File Format Distribution')
        
        # Plot 4: Resolution distribution
        if self.analysis_results.get('resolution_info'):
            resolutions = list(self.analysis_results['resolution_info'].keys())
            counts = list(self.analysis_results['resolution_info'].values())
            axes[1, 1].bar(resolutions, counts)
            axes[1, 1].set_title('Video Resolution Distribution')
            axes[1, 1].set_xlabel('Resolution')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = output_path / "bdslw60_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved to: {plot_path}")
        return str(plot_path)
    
    def save_analysis_report(self, output_dir="processed_data"):
        """Save complete analysis report as JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report_path = output_path / "analysis_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        print(f"✅ Analysis report saved to: {report_path}")
        return str(report_path)
