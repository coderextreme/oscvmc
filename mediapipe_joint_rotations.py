"""
MediaPipe Holistic Joint Rotation Calculator using Rodrigues Formula
====================================================================
This program calculates local joint rotations for hands and pose landmarks
detected by MediaPipe Holistic, using OpenCV's Rodrigues formula.

Author: Generated for MediaPipe Holistic analysis
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import time


@dataclass
class JointRotation:
    """Stores rotation information for a joint"""
    joint_name: str
    rotation_vector: np.ndarray  # Rodrigues rotation vector (3,)
    rotation_matrix: np.ndarray  # Rotation matrix (3x3)
    euler_angles: np.ndarray     # Euler angles in degrees (roll, pitch, yaw)
    parent_joint: Optional[str] = None


class MediaPipeJointRotations:
    """
    Calculates local joint rotations for MediaPipe Holistic landmarks
    using Rodrigues formula and coordinate frame transformations.
    """
    
    # Define skeletal connections for pose
    POSE_CONNECTIONS = [
        # Torso
        (11, 12),  # Left shoulder to right shoulder
        (11, 23),  # Left shoulder to left hip
        (12, 24),  # Right shoulder to right hip
        (23, 24),  # Left hip to right hip
        
        # Left arm
        (11, 13),  # Left shoulder to left elbow
        (13, 15),  # Left elbow to left wrist
        
        # Right arm
        (12, 14),  # Right shoulder to right elbow
        (14, 16),  # Right elbow to right wrist
        
        # Left leg
        (23, 25),  # Left hip to left knee
        (25, 27),  # Left knee to left ankle
        
        # Right leg
        (24, 26),  # Right hip to right knee
        (26, 28),  # Right knee to right ankle
        
        # Face connections
        (0, 1),    # Nose to left eye inner
        (0, 4),    # Nose to right eye inner
    ]
    
    # Define skeletal hierarchy for pose (parent: [children])
    POSE_HIERARCHY = {
        'root': [11, 12],  # Shoulders as root
        11: [13, 23],      # Left shoulder -> elbow, hip
        12: [14, 24],      # Right shoulder -> elbow, hip
        13: [15],          # Left elbow -> wrist
        14: [16],          # Right elbow -> wrist
        23: [25],          # Left hip -> knee
        24: [26],          # Right hip -> knee
        25: [27],          # Left knee -> ankle
        26: [28],          # Right knee -> ankle
    }
    
    # Define skeletal connections for hands
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    
    # Hand hierarchy (parent: [children])
    HAND_HIERARCHY = {
        'root': [0],  # Wrist as root
        0: [1, 5, 9, 13, 17],  # Wrist -> finger bases
        1: [2], 2: [3], 3: [4],  # Thumb
        5: [6], 6: [7], 7: [8],  # Index
        9: [10], 10: [11], 11: [12],  # Middle
        13: [14], 14: [15], 15: [16],  # Ring
        17: [18], 18: [19], 19: [20],  # Pinky
    }
    
    # Landmark names for reference
    POSE_LANDMARK_NAMES = {
        0: "nose", 11: "left_shoulder", 12: "right_shoulder",
        13: "left_elbow", 14: "right_elbow", 15: "left_wrist",
        16: "right_wrist", 23: "left_hip", 24: "right_hip",
        25: "left_knee", 26: "right_knee", 27: "left_ankle",
        28: "right_ankle"
    }
    
    def __init__(self):
        """Initialize MediaPipe Holistic and drawing utilities"""
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
    
    def normalize_vector(self, v: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length"""
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 0.0])
        return v / norm
    
    def rodrigues_to_euler(self, rotation_vector: np.ndarray) -> np.ndarray:
        """
        Convert Rodrigues rotation vector to Euler angles (roll, pitch, yaw)
        
        Args:
            rotation_vector: 3D rotation vector from Rodrigues formula
            
        Returns:
            Euler angles in degrees [roll, pitch, yaw]
        """
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Extract Euler angles from rotation matrix
        # Using XYZ convention (roll, pitch, yaw)
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = 0
        
        # Convert to degrees
        return np.degrees(np.array([roll, pitch, yaw]))
    
    def calculate_rotation_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate rotation from v1 to v2 using Rodrigues formula
        
        Args:
            v1: Initial vector (3D)
            v2: Final vector (3D)
            
        Returns:
            Tuple of (rotation_vector, rotation_matrix)
        """
        # Normalize vectors
        v1_norm = self.normalize_vector(v1)
        v2_norm = self.normalize_vector(v2)
        
        # Calculate rotation axis (cross product)
        axis = np.cross(v1_norm, v2_norm)
        axis_norm = np.linalg.norm(axis)
        
        # Calculate rotation angle
        cos_angle = np.dot(v1_norm, v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Handle parallel vectors
        if axis_norm < 1e-6:
            if cos_angle > 0:
                # Vectors are parallel and pointing same direction
                rotation_vector = np.array([0.0, 0.0, 0.0])
            else:
                # Vectors are antiparallel, need 180 degree rotation
                # Find perpendicular axis
                if abs(v1_norm[0]) < 0.9:
                    perp = np.array([1.0, 0.0, 0.0])
                else:
                    perp = np.array([0.0, 1.0, 0.0])
                axis = self.normalize_vector(np.cross(v1_norm, perp))
                rotation_vector = axis * np.pi
        else:
            # Normal case: calculate rotation vector
            axis = axis / axis_norm
            rotation_vector = axis * angle
        
        # Convert to rotation matrix using Rodrigues
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        return rotation_vector, rotation_matrix
    
    def calculate_local_joint_rotation(self, 
                                      parent_pos: np.ndarray,
                                      joint_pos: np.ndarray,
                                      child_pos: np.ndarray,
                                      reference_direction: np.ndarray = None) -> JointRotation:
        """
        Calculate local rotation at a joint using parent and child positions
        
        Args:
            parent_pos: 3D position of parent joint
            joint_pos: 3D position of current joint
            child_pos: 3D position of child joint
            reference_direction: Optional reference direction for local frame
            
        Returns:
            JointRotation object containing rotation information
        """
        # Calculate bone vectors
        bone_in = parent_pos - joint_pos  # Vector from joint to parent
        bone_out = child_pos - joint_pos  # Vector from joint to child
        
        # Use reference direction if provided, otherwise use bone_in
        if reference_direction is None:
            reference_direction = bone_in
        
        # Calculate rotation from reference to output direction
        rotation_vector, rotation_matrix = self.calculate_rotation_between_vectors(
            reference_direction, bone_out
        )
        
        # Calculate Euler angles
        euler_angles = self.rodrigues_to_euler(rotation_vector)
        
        return JointRotation(
            joint_name="joint",
            rotation_vector=rotation_vector,
            rotation_matrix=rotation_matrix,
            euler_angles=euler_angles
        )
    
    def extract_3d_landmarks(self, landmarks) -> np.ndarray:
        """
        Extract 3D coordinates from MediaPipe landmarks
        
        Args:
            landmarks: MediaPipe landmark list
            
        Returns:
            Nx3 numpy array of landmark positions
        """
        if landmarks is None:
            return None
        
        positions = []
        for landmark in landmarks.landmark:
            positions.append([landmark.x, landmark.y, landmark.z])
        
        return np.array(positions)
    
    def calculate_hand_rotations(self, hand_landmarks: np.ndarray, 
                                 hand_name: str = "hand") -> Dict[str, JointRotation]:
        """
        Calculate rotations for all joints in a hand
        
        Args:
            hand_landmarks: Nx3 array of hand landmark positions
            hand_name: Name prefix for the hand ("left_hand" or "right_hand")
            
        Returns:
            Dictionary mapping joint indices to JointRotation objects
        """
        rotations = {}
        
        if hand_landmarks is None or len(hand_landmarks) < 21:
            return rotations
        
        # Calculate rotation for each joint in the hierarchy
        for parent_idx, children in self.HAND_HIERARCHY.items():
            if parent_idx == 'root':
                continue
            
            for child_idx in children:
                # Find grandchild for calculating rotation
                if child_idx in self.HAND_HIERARCHY and len(self.HAND_HIERARCHY[child_idx]) > 0:
                    grandchild_idx = self.HAND_HIERARCHY[child_idx][0]
                    
                    parent_pos = hand_landmarks[parent_idx]
                    joint_pos = hand_landmarks[child_idx]
                    grandchild_pos = hand_landmarks[grandchild_idx]
                    
                    rotation = self.calculate_local_joint_rotation(
                        parent_pos, joint_pos, grandchild_pos
                    )
                    
                    rotation.joint_name = f"{hand_name}_joint_{child_idx}"
                    rotation.parent_joint = f"{hand_name}_joint_{parent_idx}"
                    
                    rotations[child_idx] = rotation
        
        return rotations
    
    def calculate_pose_rotations(self, pose_landmarks: np.ndarray) -> Dict[int, JointRotation]:
        """
        Calculate rotations for pose joints
        
        Args:
            pose_landmarks: Nx3 array of pose landmark positions
            
        Returns:
            Dictionary mapping joint indices to JointRotation objects
        """
        rotations = {}
        
        if pose_landmarks is None or len(pose_landmarks) < 33:
            return rotations
        
        # Calculate rotation for each joint in the hierarchy
        for parent_idx, children in self.POSE_HIERARCHY.items():
            if parent_idx == 'root':
                continue
            
            for child_idx in children:
                # Find grandchild for calculating rotation
                if child_idx in self.POSE_HIERARCHY and len(self.POSE_HIERARCHY[child_idx]) > 0:
                    grandchild_idx = self.POSE_HIERARCHY[child_idx][0]
                    
                    parent_pos = pose_landmarks[parent_idx]
                    joint_pos = pose_landmarks[child_idx]
                    grandchild_pos = pose_landmarks[grandchild_idx]
                    
                    rotation = self.calculate_local_joint_rotation(
                        parent_pos, joint_pos, grandchild_pos
                    )
                    
                    joint_name = self.POSE_LANDMARK_NAMES.get(child_idx, f"joint_{child_idx}")
                    rotation.joint_name = f"pose_{joint_name}"
                    rotation.parent_joint = f"pose_{self.POSE_LANDMARK_NAMES.get(parent_idx, f'joint_{parent_idx}')}"
                    
                    rotations[child_idx] = rotation
        
        return rotations
    
    def draw_rotation_info(self, image: np.ndarray, rotations: Dict, 
                          title: str, y_offset: int = 30) -> int:
        """
        Draw rotation information on the image
        
        Args:
            image: Image to draw on
            rotations: Dictionary of joint rotations
            title: Title for this section
            y_offset: Starting Y position
            
        Returns:
            Updated Y offset
        """
        cv2.putText(image, title, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        for idx, rotation in list(rotations.items())[:5]:  # Show first 5
            euler = rotation.euler_angles
            text = f"  {rotation.joint_name}: R={euler[0]:.1f} P={euler[1]:.1f} Y={euler[2]:.1f}"
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 20
        
        return y_offset + 10
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame and calculate all joint rotations
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (annotated_frame, all_rotations_dict)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process with MediaPipe Holistic
        results = self.holistic.process(image_rgb)
        
        # Convert back to BGR for drawing
        image_rgb.flags.writeable = True
        annotated_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Dictionary to store all rotations
        all_rotations = {
            'pose': {},
            'left_hand': {},
            'right_hand': {}
        }
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Calculate pose rotations
            pose_landmarks_3d = self.extract_3d_landmarks(results.pose_landmarks)
            all_rotations['pose'] = self.calculate_pose_rotations(pose_landmarks_3d)
        
        # Draw left hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Calculate left hand rotations
            left_hand_3d = self.extract_3d_landmarks(results.left_hand_landmarks)
            all_rotations['left_hand'] = self.calculate_hand_rotations(left_hand_3d, "left_hand")
        
        # Draw right hand landmarks
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Calculate right hand rotations
            right_hand_3d = self.extract_3d_landmarks(results.right_hand_landmarks)
            all_rotations['right_hand'] = self.calculate_hand_rotations(right_hand_3d, "right_hand")
        
        # Draw rotation information on frame
        y_offset = 30
        
        if all_rotations['pose']:
            y_offset = self.draw_rotation_info(annotated_frame, all_rotations['pose'], 
                                              "Pose Rotations:", y_offset)
        
        if all_rotations['left_hand']:
            y_offset = self.draw_rotation_info(annotated_frame, all_rotations['left_hand'], 
                                              "Left Hand Rotations:", y_offset)
        
        if all_rotations['right_hand']:
            y_offset = self.draw_rotation_info(annotated_frame, all_rotations['right_hand'], 
                                              "Right Hand Rotations:", y_offset)
        
        return annotated_frame, all_rotations
    
    def run_webcam(self):
        """Run the joint rotation calculator on webcam feed"""
        cap = cv2.VideoCapture(0)
        
        print("MediaPipe Joint Rotation Calculator")
        print("=" * 50)
        print("Press 'q' to quit")
        print("Press 's' to save current rotation data")
        print("=" * 50)
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                continue
            
            # Process frame
            annotated_frame, all_rotations = self.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            # Display FPS
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (annotated_frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('MediaPipe Joint Rotations (Rodrigues Formula)', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save rotation data
                self.save_rotation_data(all_rotations, f"rotations_frame_{frame_count}.txt")
                print(f"Saved rotation data for frame {frame_count}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_rotation_data(self, all_rotations: Dict, filename: str):
        """Save rotation data to a text file"""
        with open(filename, 'w') as f:
            f.write("MediaPipe Joint Rotations (Rodrigues Formula)\n")
            f.write("=" * 70 + "\n\n")
            
            for category, rotations in all_rotations.items():
                if rotations:
                    f.write(f"{category.upper()}\n")
                    f.write("-" * 70 + "\n")
                    
                    for idx, rotation in rotations.items():
                        f.write(f"\nJoint: {rotation.joint_name}\n")
                        f.write(f"  Parent: {rotation.parent_joint}\n")
                        f.write(f"  Rotation Vector: {rotation.rotation_vector}\n")
                        f.write(f"  Euler Angles (deg): Roll={rotation.euler_angles[0]:.2f}, "
                               f"Pitch={rotation.euler_angles[1]:.2f}, Yaw={rotation.euler_angles[2]:.2f}\n")
                        f.write(f"  Rotation Matrix:\n")
                        for row in rotation.rotation_matrix:
                            f.write(f"    [{row[0]:7.4f} {row[1]:7.4f} {row[2]:7.4f}]\n")
                    
                    f.write("\n")
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'holistic'):
            self.holistic.close()


def main():
    """Main function to run the joint rotation calculator"""
    print("\n" + "=" * 70)
    print("MediaPipe Holistic Joint Rotation Calculator")
    print("Using Rodrigues Formula for Rotation Calculation")
    print("=" * 70)
    print("\nThis program calculates local joint rotations for:")
    print("  - Pose landmarks (shoulders, elbows, knees, etc.)")
    print("  - Left hand joints (all 21 landmarks)")
    print("  - Right hand joints (all 21 landmarks)")
    print("\nRotations are calculated using:")
    print("  - Rodrigues rotation formula (OpenCV)")
    print("  - Local coordinate frames")
    print("  - Parent-joint-child relationships")
    print("=" * 70 + "\n")
    
    # Create calculator instance
    calculator = MediaPipeJointRotations()
    
    # Run on webcam
    calculator.run_webcam()
    
    print("\nThank you for using the Joint Rotation Calculator!")


if __name__ == "__main__":
    main()
