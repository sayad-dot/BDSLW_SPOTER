import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import os

class MediaPipePoseExtractor:
    """
    MediaPipe pose extraction for BdSL videos
    """

    def __init__(self):
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Pose estimation setup
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Hands estimation setup
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Face mesh setup (for head landmarks)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks_from_video(self, video_path):
        """
        Extract pose landmarks from a single video
        Returns: dictionary with frame-wise landmarks
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None

        landmarks_data = {
            'frames': [],
            'pose_landmarks': [],
            'left_hand_landmarks': [],
            'right_hand_landmarks': [],
            'face_landmarks': []
        }

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            pose_results = self.pose.process(rgb_frame)
            hands_results = self.hands.process(rgb_frame)
            face_results = self.face_mesh.process(rgb_frame)

            # Extract pose landmarks
            pose_landmarks = self._extract_pose_landmarks(pose_results)
            left_hand, right_hand = self._extract_hand_landmarks(hands_results)
            face_landmarks = self._extract_face_landmarks(face_results)

            # Store data
            landmarks_data['frames'].append(frame_count)
            landmarks_data['pose_landmarks'].append(pose_landmarks)
            landmarks_data['left_hand_landmarks'].append(left_hand)
            landmarks_data['right_hand_landmarks'].append(right_hand)  
            landmarks_data['face_landmarks'].append(face_landmarks)

            frame_count += 1

        cap.release()

        # Add metadata
        landmarks_data['metadata'] = {
            'video_path': str(video_path),
            'total_frames': frame_count,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'duration': frame_count / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }

        return landmarks_data

    def _extract_pose_landmarks(self, results):
        """Extract pose landmarks (33 body points)"""
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return landmarks
        return [0.0] * (33 * 3)  # 33 landmarks * 3 coordinates

    def _extract_hand_landmarks(self, results):
        """Extract hand landmarks (21 points per hand)"""
        left_hand = [0.0] * (21 * 3)
        right_hand = [0.0] * (21 * 3)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Determine if left or right hand
                if handedness.classification[0].label == 'Left':
                    left_hand = landmarks
                else:
                    right_hand = landmarks

        return left_hand, right_hand

    def _extract_face_landmarks(self, results):
        """Extract key face landmarks (5 points: eyes, nose, mouth corners)"""
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # Key face landmarks indices (MediaPipe face mesh)
            key_indices = [33, 263, 1, 61, 291]  # Left eye, right eye, nose tip, left mouth, right mouth
            landmarks = []
            for idx in key_indices:
                landmark = face_landmarks.landmark[idx]
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return landmarks

        return [0.0] * (5 * 3)  # 5 landmarks * 3 coordinates

    def process_dataset(self, dataset_dir, output_dir):
        """
        Process entire BdSLW60 dataset
        """
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []

        for ext in video_extensions:
            video_files.extend(list(dataset_path.rglob(f'*{ext}')))

        print(f"Found {len(video_files)} video files")

        # Process each video
        processed_data = {}

        for video_file in tqdm(video_files, desc="Processing videos"):
            video_name = video_file.stem
            landmarks_data = self.extract_landmarks_from_video(video_file)

            if landmarks_data:
                processed_data[video_name] = landmarks_data

                # Save individual file
                output_file = output_path / f"{video_name}_landmarks.json"
                with open(output_file, 'w') as f:
                    json.dump(landmarks_data, f, indent=2)

        # Save combined dataset info
        dataset_info = {
            'total_videos': len(processed_data),
            'video_names': list(processed_data.keys()),
            'processing_date': pd.Timestamp.now().isoformat()
        }

        with open(output_path / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)

        print(f"Processing complete! Saved {len(processed_data)} processed videos to {output_path}")
        return processed_data

def visualize_landmarks(landmarks_data, frame_idx=0, save_path=None):
    """
    Visualize landmarks for a specific frame
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot pose landmarks
    pose_landmarks = landmarks_data['pose_landmarks'][frame_idx]
    if pose_landmarks and any(pose_landmarks):
        pose_x = [pose_landmarks[i] for i in range(0, len(pose_landmarks), 3)]
        pose_y = [pose_landmarks[i] for i in range(1, len(pose_landmarks), 3)]
        axes[0].scatter(pose_x, pose_y, c='red', s=20)
        axes[0].set_title('Pose Landmarks')
        axes[0].invert_yaxis()

    # Plot hand landmarks  
    left_hand = landmarks_data['left_hand_landmarks'][frame_idx]
    right_hand = landmarks_data['right_hand_landmarks'][frame_idx]

    if left_hand and any(left_hand):
        left_x = [left_hand[i] for i in range(0, len(left_hand), 3)]
        left_y = [left_hand[i] for i in range(1, len(left_hand), 3)]
        axes[1].scatter(left_x, left_y, c='blue', s=20, label='Left Hand')

    if right_hand and any(right_hand):
        right_x = [right_hand[i] for i in range(0, len(right_hand), 3)]
        right_y = [right_hand[i] for i in range(1, len(right_hand), 3)]
        axes[1].scatter(right_x, right_y, c='green', s=20, label='Right Hand')

    axes[1].set_title('Hand Landmarks')
    axes[1].legend()
    axes[1].invert_yaxis()

    # Plot face landmarks
    face_landmarks = landmarks_data['face_landmarks'][frame_idx]
    if face_landmarks and any(face_landmarks):
        face_x = [face_landmarks[i] for i in range(0, len(face_landmarks), 3)]
        face_y = [face_landmarks[i] for i in range(1, len(face_landmarks), 3)]
        axes[2].scatter(face_x, face_y, c='purple', s=30)
        axes[2].set_title('Face Landmarks')
        axes[2].invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")

    plt.show()

if __name__ == "__main__":
    # Example usage
    extractor = MediaPipePoseExtractor()

    # Process sample video
    print("MediaPipe Pose Extractor initialized")
    print("To process dataset:")
    print("  extractor.process_dataset('data/', 'data/processed_landmarks/')")