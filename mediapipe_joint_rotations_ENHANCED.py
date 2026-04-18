"""
MediaPipe Holistic Joint Rotation Calculator - ENHANCED VERSION with Virtual Joints
===================================================================================
This version adds virtual joints for proper anatomical hierarchy:
- VT1: Midpoint between shoulders (cervical/thoracic junction)
- SACROILIAC: Midpoint between hips (pelvic reference)

This creates a proper spine chain: VT1 → SACROILIAC
And proper limb chains:
- Shoulders use VT1 as parent (not hips directly)
- Hips use SACROILIAC as parent (not shoulders directly)

FIXES APPLIED:
- Added virtual joints for anatomically correct hierarchy
- Proper bone hierarchies with parent-joint-child relationships
- Fixed shoulder quaternion calculation
- Enhanced pose hierarchy with spine chain
- Added quaternion output support

Author: Enhanced version with virtual joints for anatomical accuracy
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
    quaternion: np.ndarray       # Quaternion (x, y, z, w)
    parent_joint: Optional[str] = None


class MediaPipeJointRotations:
    """
    Calculates local joint rotations for MediaPipe Holistic landmarks
    using Rodrigues formula and coordinate frame transformations.
    
    ENHANCED VERSION - Uses virtual joints (VT1, SACROILIAC) for anatomically correct hierarchy.
    """
    
    # Virtual joint indices (use negative numbers to avoid conflict with real landmarks)
    VIRTUAL_VT1 = -1          # Midpoint between shoulders
    VIRTUAL_SACROILIAC = -2   # Midpoint between hips
    
    # Use official MediaPipe pose connections (35 pairs)
    POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
    
    # Use official MediaPipe hand connections (21 pairs)
    HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
    
    # ENHANCED: Define skeletal hierarchy with VIRTUAL JOINTS
    # Format: joint_idx: (parent_idx, child_idx)
    # Virtual joints create proper anatomical spine and limb references
    POSE_HIERARCHY = {
        # ========== VIRTUAL SPINE CHAIN ==========
        # VT1 (between shoulders) connects to SACROILIAC (between hips)
        VIRTUAL_VT1: (VIRTUAL_SACROILIAC, 0),  # VT1: sacroiliac -> vt1 -> nose (head)
        VIRTUAL_SACROILIAC: (VIRTUAL_VT1, 25), # Sacroiliac: vt1 -> sacroiliac -> knee (down)
        
        # ========== UPPER BODY ==========
        # Head/Neck (uses VT1 as parent)
        0: (VIRTUAL_VT1, 11),  # Nose: vt1 -> nose -> left_shoulder
        
        # Shoulders - ENHANCED: Now use VT1 as parent (proper cervical reference)
        11: (VIRTUAL_VT1, 13),  # Left shoulder: vt1 -> shoulder -> elbow
        12: (VIRTUAL_VT1, 14),  # Right shoulder: vt1 -> shoulder -> elbow
        
        # Elbows
        13: (11, 15),  # Left elbow: shoulder -> elbow -> wrist
        14: (12, 16),  # Right elbow: shoulder -> elbow -> wrist
        
        # Wrists
        15: (13, 19),  # Left wrist: elbow -> wrist -> index
        16: (14, 20),  # Right wrist: elbow -> wrist -> index
        
        # ========== LOWER BODY ==========
        # Hips - ENHANCED: Now use SACROILIAC as parent (proper pelvic reference)
        23: (VIRTUAL_SACROILIAC, 25),  # Left hip: sacroiliac -> hip -> knee
        24: (VIRTUAL_SACROILIAC, 26),  # Right hip: sacroiliac -> hip -> knee
        
        # Knees
        25: (23, 27),  # Left knee: hip -> knee -> ankle
        26: (24, 28),  # Right knee: hip -> knee -> ankle
        
        # Ankles
        27: (25, 31),  # Left ankle: knee -> ankle -> foot_index
        28: (26, 32),  # Right ankle: knee -> ankle -> foot_index
        
        # Feet
        31: (27, 29),  # Left foot_index: ankle -> foot_index -> heel
        32: (28, 30),  # Right foot_index: ankle -> foot_index -> heel
    }
    
    # Hand hierarchy (unchanged from previous version)
    HAND_HIERARCHY = {
        # Thumb chain
        1: (0, 2),    # Thumb CMC: wrist -> cmc -> mcp
        2: (1, 3),    # Thumb MCP: cmc -> mcp -> ip
        3: (2, 4),    # Thumb IP: mcp -> ip -> tip
        
        # Index finger chain
        5: (0, 6),    # Index MCP: wrist -> mcp -> pip
        6: (5, 7),    # Index PIP: mcp -> pip -> dip
        7: (6, 8),    # Index DIP: pip -> dip -> tip
        
        # Middle finger chain
        9: (0, 10),   # Middle MCP: wrist -> mcp -> pip
        10: (9, 11),  # Middle PIP: mcp -> pip -> dip
        11: (10, 12), # Middle DIP: pip -> dip -> tip
        
        # Ring finger chain
        13: (0, 14),  # Ring MCP: wrist -> mcp -> pip
        14: (13, 15), # Ring PIP: mcp -> pip -> dip
        15: (14, 16), # Ring DIP: pip -> dip -> tip
        
        # Pinky chain
        17: (0, 18),  # Pinky MCP: wrist -> mcp -> pip
        18: (17, 19), # Pinky PIP: mcp -> pip -> dip
        19: (18, 20), # Pinky DIP: pip -> dip -> tip
        
        # Wrist (uses palm orientation)
        0: (9, 1),    # Wrist: middle_mcp -> wrist -> thumb_cmc
    }
    
    # Landmark names for reference
    POSE_LANDMARK_NAMES = {
        -1: "vt1", -2: "sacroiliac",
        0: "nose", 11: "left_shoulder", 12: "right_shoulder",
        13: "left_elbow", 14: "right_elbow", 15: "left_wrist",
        16: "right_wrist", 23: "left_hip", 24: "right_hip",
        25: "left_knee", 26: "right_knee", 27: "left_ankle",
        28: "right_ankle", 31: "left_foot_index", 32: "right_foot_index"
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
    
    def calculate_virtual_joints(self, pose_landmarks: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Calculate virtual joint positions from real landmarks.
        
        Args:
            pose_landmarks: Nx3 array of pose landmark positions
            
        Returns:
            Dictionary mapping virtual joint indices to 3D positions
        """
        virtual_joints = {}
        
        if pose_landmarks is None or len(pose_landmarks) < 33:
            return virtual_joints
        
        try:
            # VT1: Midpoint between left and right shoulders
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            vt1 = (left_shoulder + right_shoulder) / 2.0
            virtual_joints[self.VIRTUAL_VT1] = vt1
            
            # SACROILIAC: Midpoint between left and right hips
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
            sacroiliac = (left_hip + right_hip) / 2.0
            virtual_joints[self.VIRTUAL_SACROILIAC] = sacroiliac
            
        except (IndexError, KeyError):
            # If landmarks are missing, return empty dict
            pass
        
        return virtual_joints
    
    def normalize_vector(self, v: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length"""
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 0.0])
        return v / norm
    
    def rotation_matrix_to_quaternion(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion [x, y, z, w]
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Quaternion as [x, y, z, w]
        """
        if rotation_matrix.shape != (3, 3):
            return np.array([0.0, 0.0, 0.0, 1.0])
        
        trace = np.trace(rotation_matrix)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
        elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])
    
    def rodrigues_to_euler(self, rotation_vector: np.ndarray) -> np.ndarray:
        """
        Convert Rodrigues rotation vector to Euler angles (roll, pitch, yaw)
        
        Args:
            rotation_vector: 3D rotation vector from Rodrigues formula
            
        Returns:
            Euler angles in degrees [roll, pitch, yaw]
        """
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
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
        v1_norm = self.normalize_vector(v1)
        v2_norm = self.normalize_vector(v2)
        
        axis = np.cross(v1_norm, v2_norm)
        axis_norm = np.linalg.norm(axis)
        
        cos_angle = np.dot(v1_norm, v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        if axis_norm < 1e-6:
            if cos_angle > 0:
                rotation_vector = np.array([0.0, 0.0, 0.0])
            else:
                if abs(v1_norm[0]) < 0.9:
                    perp = np.array([1.0, 0.0, 0.0])
                else:
                    perp = np.array([0.0, 1.0, 0.0])
                axis = self.normalize_vector(np.cross(v1_norm, perp))
                rotation_vector = axis * np.pi
        else:
            axis = axis / axis_norm
            rotation_vector = axis * angle
        
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
        bone_in = parent_pos - joint_pos
        bone_out = child_pos - joint_pos
        
        if reference_direction is None:
            reference_direction = bone_in
        
        rotation_vector, rotation_matrix = self.calculate_rotation_between_vectors(
            reference_direction, bone_out
        )
        
        euler_angles = self.rodrigues_to_euler(rotation_vector)
        quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
        
        return JointRotation(
            joint_name="joint",
            rotation_vector=rotation_vector,
            rotation_matrix=rotation_matrix,
            euler_angles=euler_angles,
            quaternion=quaternion
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
                                 hand_name: str = "hand") -> Dict[int, JointRotation]:
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
        
        for joint_idx, (parent_idx, child_idx) in self.HAND_HIERARCHY.items():
            try:
                parent_pos = hand_landmarks[parent_idx]
                joint_pos = hand_landmarks[joint_idx]
                child_pos = hand_landmarks[child_idx]
                
                rotation = self.calculate_local_joint_rotation(
                    parent_pos, joint_pos, child_pos
                )
                
                rotation.joint_name = f"{hand_name}_joint_{joint_idx}"
                rotation.parent_joint = f"{hand_name}_joint_{parent_idx}"
                
                rotations[joint_idx] = rotation
            except (IndexError, KeyError):
                continue
        
        return rotations
    
    def calculate_pose_rotations(self, pose_landmarks: np.ndarray) -> Dict[int, JointRotation]:
        """
        Calculate rotations for pose joints using ENHANCED hierarchy with virtual joints.
        
        This now includes VT1 (between shoulders) and SACROILIAC (between hips)
        for anatomically correct spine and limb rotations!
        
        Args:
            pose_landmarks: Nx3 array of pose landmark positions
            
        Returns:
            Dictionary mapping joint indices to JointRotation objects
        """
        rotations = {}
        
        if pose_landmarks is None or len(pose_landmarks) < 33:
            return rotations
        
        # Calculate virtual joint positions
        virtual_joints = self.calculate_virtual_joints(pose_landmarks)
        
        # Combine real and virtual landmarks
        all_landmarks = {}
        for i in range(len(pose_landmarks)):
            all_landmarks[i] = pose_landmarks[i]
        all_landmarks.update(virtual_joints)
        
        # Calculate rotations using enhanced hierarchy
        for joint_idx, (parent_idx, child_idx) in self.POSE_HIERARCHY.items():
            try:
                # Get positions (handles both real and virtual joints)
                parent_pos = all_landmarks[parent_idx]
                joint_pos = all_landmarks[joint_idx]
                child_pos = all_landmarks[child_idx]
                
                rotation = self.calculate_local_joint_rotation(
                    parent_pos, joint_pos, child_pos
                )
                
                joint_name = self.POSE_LANDMARK_NAMES.get(joint_idx, f"joint_{joint_idx}")
                rotation.joint_name = f"pose_{joint_name}"
                rotation.parent_joint = f"pose_{self.POSE_LANDMARK_NAMES.get(parent_idx, f'joint_{parent_idx}')}"
                
                rotations[joint_idx] = rotation
            except (IndexError, KeyError):
                continue
        
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
        
        # Show key joints including virtual ones
        key_joints = [-1, -2, 11, 12, 13, 14, 23, 24]  # vt1, sacroiliac, shoulders, elbows, hips
        display_count = 0
        
        for idx in key_joints:
            if idx in rotations and display_count < 8:
                rotation = rotations[idx]
                euler = rotation.euler_angles
                quat = rotation.quaternion
                
                text1 = f"  {rotation.joint_name}: R={euler[0]:.1f} P={euler[1]:.1f} Y={euler[2]:.1f}"
                text2 = f"    Q=[{quat[0]:.2f},{quat[1]:.2f},{quat[2]:.2f},{quat[3]:.2f}]"
                
                cv2.putText(image, text1, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y_offset += 15
                
                if np.allclose(quat, [0, 0, 0, 1], atol=1e-3):
                    cv2.putText(image, text2 + " [IDENTITY!]", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                else:
                    cv2.putText(image, text2, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 100), 1)
                y_offset += 20
                display_count += 1
        
        return y_offset + 10
    
    def draw_virtual_joints(self, image: np.ndarray, pose_landmarks, 
                           virtual_joints: Dict[int, np.ndarray]):
        """
        Draw virtual joints on the image
        
        Args:
            image: Image to draw on
            pose_landmarks: MediaPipe pose landmarks
            virtual_joints: Dictionary of virtual joint positions
        """
        if pose_landmarks is None:
            return
        
        h, w, _ = image.shape
        
        # Draw VT1 (between shoulders)
        if self.VIRTUAL_VT1 in virtual_joints:
            vt1 = virtual_joints[self.VIRTUAL_VT1]
            x, y = int(vt1[0] * w), int(vt1[1] * h)
            cv2.circle(image, (x, y), 8, (255, 0, 255), -1)  # Magenta
            cv2.putText(image, "VT1", (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Draw SACROILIAC (between hips)
        if self.VIRTUAL_SACROILIAC in virtual_joints:
            sacro = virtual_joints[self.VIRTUAL_SACROILIAC]
            x, y = int(sacro[0] * w), int(sacro[1] * h)
            cv2.circle(image, (x, y), 8, (255, 255, 0), -1)  # Cyan
            cv2.putText(image, "SACRO", (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw spine connection (VT1 to SACROILIAC)
        if self.VIRTUAL_VT1 in virtual_joints and self.VIRTUAL_SACROILIAC in virtual_joints:
            vt1 = virtual_joints[self.VIRTUAL_VT1]
            sacro = virtual_joints[self.VIRTUAL_SACROILIAC]
            x1, y1 = int(vt1[0] * w), int(vt1[1] * h)
            x2, y2 = int(sacro[0] * w), int(sacro[1] * h)
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow spine line
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame and calculate all joint rotations
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (annotated_frame, all_rotations_dict)
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = self.holistic.process(image_rgb)
        
        image_rgb.flags.writeable = True
        annotated_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        all_rotations = {
            'pose': {},
            'left_hand': {},
            'right_hand': {}
        }
        
        virtual_joints = {}
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            pose_landmarks_3d = self.extract_3d_landmarks(results.pose_landmarks)
            virtual_joints = self.calculate_virtual_joints(pose_landmarks_3d)
            all_rotations['pose'] = self.calculate_pose_rotations(pose_landmarks_3d)
            
            # Draw virtual joints on frame
            self.draw_virtual_joints(annotated_frame, results.pose_landmarks, virtual_joints)
        
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            left_hand_3d = self.extract_3d_landmarks(results.left_hand_landmarks)
            all_rotations['left_hand'] = self.calculate_hand_rotations(left_hand_3d, "left_hand")
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            right_hand_3d = self.extract_3d_landmarks(results.right_hand_landmarks)
            all_rotations['right_hand'] = self.calculate_hand_rotations(right_hand_3d, "right_hand")
        
        y_offset = 30
        
        if all_rotations['pose']:
            y_offset = self.draw_rotation_info(annotated_frame, all_rotations['pose'], 
                                              "Pose Rotations (Virtual Joints):", y_offset)
        
        if all_rotations['left_hand']:
            y_offset = self.draw_rotation_info(annotated_frame, all_rotations['left_hand'], 
                                              "Left Hand:", y_offset)
        
        if all_rotations['right_hand']:
            y_offset = self.draw_rotation_info(annotated_frame, all_rotations['right_hand'], 
                                              "Right Hand:", y_offset)
        
        return annotated_frame, all_rotations
    
    def run_webcam(self):
        """Run the joint rotation calculator on webcam feed"""
        cap = cv2.VideoCapture(0)
        
        print("MediaPipe Joint Rotation Calculator - ENHANCED with Virtual Joints")
        print("=" * 70)
        print("Press 'q' to quit")
        print("Press 's' to save current rotation data")
        print("=" * 70)
        print("\nVIRTUAL JOINTS ADDED:")
        print("  ✓ VT1: Midpoint between shoulders (cervical/thoracic junction)")
        print("  ✓ SACROILIAC: Midpoint between hips (pelvic reference)")
        print("  ✓ SPINE CHAIN: VT1 → SACROILIAC (proper anatomical axis)")
        print("\nHIERARCHY IMPROVEMENTS:")
        print("  ✓ Shoulders now use VT1 as parent (not hips)")
        print("  ✓ Hips now use SACROILIAC as parent (not shoulders)")
        print("  ✓ Anatomically correct kinematic chain")
        print("=" * 70)
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                continue
            
            annotated_frame, all_rotations = self.process_frame(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (annotated_frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('MediaPipe - ENHANCED with VT1 & SACROILIAC', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_rotation_data(all_rotations, f"rotations_frame_{frame_count}.txt")
                print(f"Saved rotation data for frame {frame_count}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_rotation_data(self, all_rotations: Dict, filename: str):
        """Save rotation data to a text file"""
        with open(filename, 'w') as f:
            f.write("MediaPipe Joint Rotations - ENHANCED with Virtual Joints\n")
            f.write("Virtual Joints: VT1 (shoulders midpoint), SACROILIAC (hips midpoint)\n")
            f.write("=" * 70 + "\n\n")
            
            for category, rotations in all_rotations.items():
                if rotations:
                    f.write(f"{category.upper()}\n")
                    f.write("-" * 70 + "\n")
                    
                    for idx, rotation in rotations.items():
                        f.write(f"\nJoint: {rotation.joint_name}\n")
                        f.write(f"  Parent: {rotation.parent_joint}\n")
                        f.write(f"  Rotation Vector: {rotation.rotation_vector}\n")
                        f.write(f"  Quaternion (x,y,z,w): [{rotation.quaternion[0]:.4f}, "
                               f"{rotation.quaternion[1]:.4f}, {rotation.quaternion[2]:.4f}, "
                               f"{rotation.quaternion[3]:.4f}]\n")
                        
                        if np.allclose(rotation.quaternion, [0, 0, 0, 1], atol=1e-3):
                            f.write(f"    ⚠️  WARNING: Identity quaternion detected!\n")
                        
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
    print("ENHANCED VERSION with Virtual Joints")
    print("=" * 70)
    print("\nVirtual Joints for Anatomical Accuracy:")
    print("  • VT1: Vertebra between shoulders (C7/T1 junction)")
    print("  • SACROILIAC: Joint between hips (pelvic reference)")
    print("\nProper Kinematic Chains:")
    print("  • Shoulders → VT1 → SACROILIAC → Hips")
    print("  • Arms: VT1 → Shoulder → Elbow → Wrist")
    print("  • Legs: SACROILIAC → Hip → Knee → Ankle")
    print("\nThis creates anatomically correct rotations for:")
    print("  ✓ Shoulder movements (now relative to spine, not hips)")
    print("  ✓ Hip movements (now relative to pelvis, not shoulders)")
    print("  ✓ Spine bending (VT1 to SACROILIAC connection)")
    print("  ✓ All other joints with proper parent references")
    print("=" * 70 + "\n")
    
    calculator = MediaPipeJointRotations()
    calculator.run_webcam()
    
    print("\nThank you for using the ENHANCED Joint Rotation Calculator!")


if __name__ == "__main__":
    main()
