import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from pathlib import Path
import json

class SPOTERPoseNormalizer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        
        # SPOTER normalization parameters
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS
        self.hand_connections = self.mp_hands.HAND_CONNECTIONS
        
        # Signing space parameters (adapted for BdSL)
        self.signing_space = {
            'width_ratio': 0.85,  # BdSL may use slightly more horizontal space
            'height_ratio': 0.9,
            'center_x': 0.5,
            'center_y': 0.45  # Slightly higher center for BdSL signing
        }
        
        # Initialize for temporal smoothing
        self.previous_points = None
        
    def normalize_pose_sequence(self, pose_landmarks_sequence):
        """
        Apply SPOTER-style normalization to pose landmark sequence
        
        Args:
            pose_landmarks_sequence: List of pose landmarks for each frame
            
        Returns:
            normalized_sequence: Normalized pose sequence ready for SPOTER
        """
        if not pose_landmarks_sequence:
            return []
            
        normalized_frames = []
        
        for frame_landmarks in pose_landmarks_sequence:
            if frame_landmarks is None:
                continue
                
            # Extract key landmarks
            normalized_frame = self._normalize_single_frame(frame_landmarks)
            if normalized_frame is not None:
                normalized_frames.append(normalized_frame)
        
        return normalized_frames
    
    def _normalize_single_frame(self, landmarks):
        """Normalize landmarks for a single frame using SPOTER methodology"""
        try:
            # Handle None landmarks
            if landmarks is None:
                return None
            
            # Convert landmarks to numpy array
            if hasattr(landmarks, 'landmark'):
                # MediaPipe format
                points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            elif isinstance(landmarks, np.ndarray):
                # Already numpy array
                points = landmarks
            elif isinstance(landmarks, list):
                # Check if it's a flat list (needs reshaping)
                if len(landmarks) == 99:  # 33 * 3 coordinates
                    points = np.array(landmarks).reshape(33, 3)
                else:
                    # List format
                    points = np.array(landmarks)
            else:
                print(f"Unknown landmark format: {type(landmarks)}")
                return None
            
            # Ensure proper shape
            if len(points.shape) != 2 or points.shape[1] != 3:
                print(f"Warning: Expected 3D coordinates, got shape {points.shape}")
                return None
            
            # SPOTER normalization steps:
            # 1. Torso-based normalization
            normalized_points = self._torso_normalization(points)
            
            # 2. Signing space normalization  
            normalized_points = self._signing_space_normalization(normalized_points)
            
            # 3. Temporal smoothing (if needed)
            normalized_points = self._smooth_landmarks(normalized_points)
            
            return normalized_points
            
        except Exception as e:
            print(f"Error normalizing frame: {e}")
            return None
    
    def _torso_normalization(self, points):
        """Normalize based on torso landmarks (SPOTER approach)"""
        try:
            # Ensure points is a numpy array
            if not isinstance(points, np.ndarray):
                points = np.array(points)
            
            # Check if we have enough landmarks (33 for pose)
            if len(points) < 25:
                print(f"Warning: Insufficient landmarks ({len(points)}), skipping normalization")
                return points
            
            # Key torso landmarks for normalization
            left_shoulder = points[11]   # LEFT_SHOULDER
            right_shoulder = points[12]  # RIGHT_SHOULDER
            left_hip = points[23]        # LEFT_HIP  
            right_hip = points[24]       # RIGHT_HIP
            
            # Ensure landmarks are 3D coordinates
            if left_shoulder.shape != (3,) or right_shoulder.shape != (3,):
                print("Warning: Invalid landmark format, skipping normalization")
                return points
            
            # Calculate torso center and scale
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            torso_center = (shoulder_center + hip_center) / 2
            
            # Torso scale (shoulder width)
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            
            # Normalize all points relative to torso
            if shoulder_width > 0.01:  # Avoid division by very small numbers
                normalized_points = (points - torso_center) / shoulder_width
            else:
                normalized_points = points - torso_center
                
            return normalized_points
            
        except Exception as e:
            print(f"Error in torso normalization: {e}")
            return points
    
    def _signing_space_normalization(self, points):
        """Apply signing space constraints (BdSL-specific)"""
        try:
            # Scale to signing space dimensions
            points[:, 0] *= self.signing_space['width_ratio']
            points[:, 1] *= self.signing_space['height_ratio']
            
            # Center in signing space
            points[:, 0] += self.signing_space['center_x'] - 0.5
            points[:, 1] += self.signing_space['center_y'] - 0.5
            
            return points
            
        except Exception:
            return points
    
    def _smooth_landmarks(self, points, alpha=0.7):
        """Apply temporal smoothing to reduce jitter"""
        # Simple exponential smoothing
        if hasattr(self, 'previous_points') and self.previous_points is not None:
            smoothed_points = alpha * points + (1 - alpha) * self.previous_points
        else:
            smoothed_points = points
            
        self.previous_points = smoothed_points.copy()
        return smoothed_points
    
    def create_combined_feature_vector(self, pose_landmarks, hand_landmarks_left=None, hand_landmarks_right=None):
        """
        Create SPOTER-compatible feature vector combining pose, hands
        
        Returns:
            feature_vector: 108-dimensional feature vector (36 pose + 36 left hand + 36 right hand)
        """
        feature_vector = []
        
        # Pose features (36 dimensions: 12 key joints * 3 coordinates)
        pose_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # Upper body + arms
        
        if pose_landmarks is not None:
            for idx in pose_indices:
                if idx < len(pose_landmarks):
                    point = pose_landmarks[idx]
                    feature_vector.extend([point[0], point[1], point[2]])
                else:
                    feature_vector.extend([0.0, 0.0, 0.0])
        else:
            feature_vector.extend([0.0] * 36)
        
        # Left hand features (36 dimensions: 12 key points * 3 coordinates)
        hand_indices = [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 2]  # Key hand landmarks
        
        if hand_landmarks_left is not None:
            for idx in hand_indices:
                if idx < len(hand_landmarks_left):
                    point = hand_landmarks_left[idx]
                    feature_vector.extend([point[0], point[1], point[2]])
                else:
                    feature_vector.extend([0.0, 0.0, 0.0])
        else:
            feature_vector.extend([0.0] * 36)
        
        # Right hand features (36 dimensions)
        if hand_landmarks_right is not None:
            for idx in hand_indices:
                if idx < len(hand_landmarks_right):
                    point = hand_landmarks_right[idx]
                    feature_vector.extend([point[0], point[1], point[2]])
                else:
                    feature_vector.extend([0.0, 0.0, 0.0])
        else:
            feature_vector.extend([0.0] * 36)
        
        return np.array(feature_vector)
    
    def visualize_normalization(self, original_landmarks, normalized_landmarks, output_path="processed_data"):
        """Visualize before/after normalization"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original landmarks
            if len(original_landmarks) > 0:
                orig_points = np.array(original_landmarks)
                ax1.scatter(orig_points[:, 0], orig_points[:, 1], alpha=0.6)
                ax1.set_title('Original Landmarks')
                ax1.set_aspect('equal')
                ax1.grid(True)
            
            # Normalized landmarks  
            if len(normalized_landmarks) > 0:
                norm_points = np.array(normalized_landmarks)
                ax2.scatter(norm_points[:, 0], norm_points[:, 1], alpha=0.6, color='red')
                ax2.set_title('Normalized Landmarks (SPOTER-style)')
                ax2.set_aspect('equal')
                ax2.grid(True)
            
            plt.tight_layout()
            
            # Save visualization
            Path(output_path).mkdir(exist_ok=True)
            viz_path = Path(output_path) / "normalization_comparison.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Normalization visualization saved to: {viz_path}")
            return str(viz_path)
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
