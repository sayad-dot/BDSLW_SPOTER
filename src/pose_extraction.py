import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import json


class MediaPipePoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe models
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_pose_from_video(self, video_path):
        """
        Extract pose landmarks from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict: Contains frames data, pose_landmarks, hand_landmarks
        """
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"❌ Video file not found: {video_path}")
            return {'frames': [], 'pose_landmarks': [], 'hand_landmarks': []}
        
        print(f"🎥 Processing video: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return {'frames': [], 'pose_landmarks': [], 'hand_landmarks': []}
        
        frames_data = []
        pose_landmarks_sequence = []
        hand_landmarks_sequence = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            pose_results = self.pose.process(rgb_frame)
            hand_results = self.hands.process(rgb_frame)
            
            # Extract pose landmarks
            pose_landmarks = None
            if pose_results.pose_landmarks:
                # Convert to numpy array format
                pose_landmarks = np.array([
                    [landmark.x, landmark.y, landmark.z] 
                    for landmark in pose_results.pose_landmarks.landmark
                ])
            
            # Extract hand landmarks
            hand_landmarks = {'left': None, 'right': None}
            if hand_results.multi_hand_landmarks:
                for idx, hand_landmark in enumerate(hand_results.multi_hand_landmarks):
                    # Convert to numpy array
                    hand_array = np.array([
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in hand_landmark.landmark
                    ])
                    
                    # Determine left/right hand
                    if hand_results.multi_handedness:
                        handedness = hand_results.multi_handedness[idx].classification[0].label
                        if handedness == 'Left':
                            hand_landmarks['left'] = hand_array
                        else:
                            hand_landmarks['right'] = hand_array
            
            # Store frame data
            frames_data.append({
                'frame_number': frame_count,
                'height': frame.shape[0],
                'width': frame.shape[1],
                'has_pose': pose_landmarks is not None,
                'has_hands': hand_landmarks['left'] is not None or hand_landmarks['right'] is not None
            })
            
            pose_landmarks_sequence.append(pose_landmarks)
            hand_landmarks_sequence.append(hand_landmarks)
        
        cap.release()
        
        print(f"✓ Extracted {frame_count} frames")
        
        return {
            'frames': frames_data,
            'pose_landmarks': pose_landmarks_sequence,
            'hand_landmarks': hand_landmarks_sequence,
            'video_info': {
                'total_frames': frame_count,
                'video_path': str(video_path),
                'video_name': video_path.name
            }
        }
    
    def save_landmarks(self, landmarks_data, output_path):
        """Save extracted landmarks to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            'frames': landmarks_data['frames'],
            'video_info': landmarks_data['video_info'],
            'pose_landmarks': [],
            'hand_landmarks': []
        }
        
        # Convert pose landmarks
        for pose_data in landmarks_data['pose_landmarks']:
            if pose_data is not None:
                serializable_data['pose_landmarks'].append(pose_data.tolist())
            else:
                serializable_data['pose_landmarks'].append(None)
        
        # Convert hand landmarks
        for hand_data in landmarks_data['hand_landmarks']:
            hand_serializable = {}
            if hand_data['left'] is not None:
                hand_serializable['left'] = hand_data['left'].tolist()
            else:
                hand_serializable['left'] = None
                
            if hand_data['right'] is not None:
                hand_serializable['right'] = hand_data['right'].tolist()
            else:
                hand_serializable['right'] = None
                
            serializable_data['hand_landmarks'].append(hand_serializable)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"✅ Landmarks saved to: {output_path}")
        return str(output_path)
