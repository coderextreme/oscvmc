"""
Enhanced OSC/VMC Integration with Virtual Joints (VT1 & SACROILIAC)
===================================================================

This module extends the OSC/VMC streaming to include virtual joints:
- VT1 (between shoulders) - spine upper reference
- SACROILIAC (between hips) - spine lower reference

These virtual joints provide proper spine axis representation in VR/VMC applications.
"""

import numpy as np
from pythonosc import udp_client
import time
from typing import Dict


class EnhancedOSCVMCClient:
    """
    Enhanced OSC/VMC client that includes virtual joints (VT1, SACROILIAC)
    Compatible with standard VMC protocol while adding spine representation
    """
    
    # Virtual joint indices
    VIRTUAL_VT1 = -1
    VIRTUAL_SACROILIAC = -2
    
    # Enhanced VMC bone mapping including virtual joints
    VMC_BONE_MAPPING = {
        'pose': {
            # Virtual joints (NEW!)
            -1: 'Spine',           # VT1 - upper spine (between shoulders)
            -2: 'Hips',            # SACROILIAC - pelvis center (between hips)
            
            # Standard pose bones
            0: 'Head',
            11: 'LeftShoulder',
            12: 'RightShoulder',
            13: 'LeftUpperArm',
            14: 'RightUpperArm',
            15: 'LeftLowerArm',
            16: 'RightLowerArm',
            23: 'LeftUpperLeg',
            24: 'RightUpperLeg',
            25: 'LeftLowerLeg',
            26: 'RightLowerLeg',
            27: 'LeftFoot',
            28: 'RightFoot',
        },
        'left_hand': {
            0: 'LeftHand',
            1: 'LeftThumbProximal',
            2: 'LeftThumbIntermediate',
            3: 'LeftThumbDistal',
            5: 'LeftIndexProximal',
            6: 'LeftIndexIntermediate',
            7: 'LeftIndexDistal',
            9: 'LeftMiddleProximal',
            10: 'LeftMiddleIntermediate',
            11: 'LeftMiddleDistal',
            13: 'LeftRingProximal',
            14: 'LeftRingIntermediate',
            15: 'LeftRingDistal',
            17: 'LeftLittleProximal',
            18: 'LeftLittleIntermediate',
            19: 'LeftLittleDistal',
        },
        'right_hand': {
            0: 'RightHand',
            1: 'RightThumbProximal',
            2: 'RightThumbIntermediate',
            3: 'RightThumbDistal',
            5: 'RightIndexProximal',
            6: 'RightIndexIntermediate',
            7: 'RightIndexDistal',
            9: 'RightMiddleProximal',
            10: 'RightMiddleIntermediate',
            11: 'RightMiddleDistal',
            13: 'RightRingProximal',
            14: 'RightRingIntermediate',
            15: 'RightRingDistal',
            17: 'RightLittleProximal',
            18: 'RightLittleIntermediate',
            19: 'RightLittleDistal',
        }
    }
    
    def __init__(self, ip: str = "127.0.0.1", port: int = 39539, 
                 include_virtual_joints: bool = True):
        """
        Initialize Enhanced OSC/VMC client
        
        Args:
            ip: Target IP address
            port: Target port (default: 39539 - VMC standard)
            include_virtual_joints: If True, send VT1 and SACROILIAC (default: True)
        """
        self.ip = ip
        self.port = port
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.frame_count = 0
        self.start_time = time.time()
        self.include_virtual_joints = include_virtual_joints
        
        print(f"Enhanced OSC/VMC Client initialized: {ip}:{port}")
        print("VMC Protocol Address Patterns:")
        print("  /VMC/Ext/Bone/Pos - Bone position and rotation")
        print("  /VMC/Ext/Root/Pos - Root position and rotation")
        print("  /VMC/Ext/OK - Frame available signal")
        print("  /VMC/Ext/T - Time signal")
        if include_virtual_joints:
            print("\n✨ ENHANCED: Virtual joints enabled!")
            print("  VT1 (Spine) - Upper spine reference")
            print("  SACROILIAC (Hips) - Pelvic reference")
    
    def quaternion_from_rotation_matrix(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion (x, y, z, w)
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion as [x, y, z, w] (VMC convention)
        """
        if R.shape != (3, 3):
            return np.array([0.0, 0.0, 0.0, 1.0])
        
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])
    
    def send_bone_transform(self, bone_name: str, position: np.ndarray, 
                           rotation_matrix: np.ndarray = None,
                           quaternion: np.ndarray = None):
        """
        Send bone position and rotation via VMC protocol
        
        Args:
            bone_name: Name of the bone (VMC bone naming)
            position: 3D position [x, y, z]
            rotation_matrix: 3x3 rotation matrix (optional if quaternion provided)
            quaternion: [x, y, z, w] quaternion (optional if rotation_matrix provided)
        """
        # Get quaternion from either source
        if quaternion is not None:
            quat = quaternion
        elif rotation_matrix is not None:
            quat = self.quaternion_from_rotation_matrix(rotation_matrix)
        else:
            quat = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        
        # VMC protocol: /VMC/Ext/Bone/Pos
        # Format: bone_name, px, py, pz, qx, qy, qz, qw
        address = "/VMC/Ext/Bone/Pos"
        
        self.client.send_message(address, [
            bone_name,
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(quat[0]),  # qx
            float(quat[1]),  # qy
            float(quat[2]),  # qz
            float(quat[3])   # qw
        ])
    
    def send_root_transform(self, position: np.ndarray, 
                           rotation_matrix: np.ndarray = None,
                           quaternion: np.ndarray = None):
        """
        Send root position and rotation via VMC protocol
        
        Args:
            position: 3D position [x, y, z]
            rotation_matrix: 3x3 rotation matrix (optional)
            quaternion: [x, y, z, w] quaternion (optional)
        """
        if quaternion is not None:
            quat = quaternion
        elif rotation_matrix is not None:
            quat = self.quaternion_from_rotation_matrix(rotation_matrix)
        else:
            quat = np.array([0.0, 0.0, 0.0, 1.0])
        
        address = "/VMC/Ext/Root/Pos"
        
        self.client.send_message(address, [
            "root",
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
            0.0, 0.0, 0.0, 0.0  # Scale (not used)
        ])
    
    def send_time(self):
        """Send time signal (frame timing)"""
        current_time = time.time() - self.start_time
        self.client.send_message("/VMC/Ext/T", [float(current_time)])
    
    def send_frame_available(self):
        """Send frame available signal"""
        self.client.send_message("/VMC/Ext/OK", [1])
        self.frame_count += 1
    
    def calculate_virtual_joint_positions(self, landmarks_3d: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Calculate virtual joint positions from real landmarks
        
        Args:
            landmarks_3d: Array of 3D pose landmarks
            
        Returns:
            Dictionary mapping virtual joint indices to positions
        """
        virtual_positions = {}
        
        if landmarks_3d is None or len(landmarks_3d) < 33:
            return virtual_positions
        
        try:
            # VT1: Midpoint between shoulders
            left_shoulder = landmarks_3d[11]
            right_shoulder = landmarks_3d[12]
            vt1 = (left_shoulder + right_shoulder) / 2.0
            virtual_positions[self.VIRTUAL_VT1] = vt1
            
            # SACROILIAC: Midpoint between hips
            left_hip = landmarks_3d[23]
            right_hip = landmarks_3d[24]
            sacroiliac = (left_hip + right_hip) / 2.0
            virtual_positions[self.VIRTUAL_SACROILIAC] = sacroiliac
            
        except (IndexError, KeyError):
            pass
        
        return virtual_positions
    
    def send_all_bones(self, landmarks_3d: dict, rotations: dict):
        """
        Send all bone transforms including virtual joints
        
        Args:
            landmarks_3d: Dictionary of landmark positions by category
            rotations: Dictionary of joint rotations by category
        """
        # Send time signal
        self.send_time()
        
        # Process pose landmarks
        if 'pose' in landmarks_3d and 'pose' in rotations:
            pose_landmarks = landmarks_3d['pose']
            pose_rotations = rotations['pose']
            
            # Calculate virtual joint positions
            if self.include_virtual_joints:
                virtual_positions = self.calculate_virtual_joint_positions(pose_landmarks)
            else:
                virtual_positions = {}
            
            # Send all pose bones (including virtual joints if enabled)
            for landmark_idx, bone_name in self.VMC_BONE_MAPPING['pose'].items():
                position = None
                rotation_matrix = None
                quaternion = None
                
                # Handle virtual joints
                if landmark_idx < 0:  # Virtual joint
                    if not self.include_virtual_joints:
                        continue  # Skip if virtual joints disabled
                    
                    if landmark_idx in virtual_positions:
                        position = virtual_positions[landmark_idx]
                        
                        # Get rotation from rotations dict if available
                        if landmark_idx in pose_rotations:
                            rotation = pose_rotations[landmark_idx]
                            rotation_matrix = rotation.rotation_matrix
                            # Use quaternion if available (ENHANCED version)
                            if hasattr(rotation, 'quaternion'):
                                quaternion = rotation.quaternion
                        else:
                            rotation_matrix = np.eye(3)  # Identity if no rotation
                
                # Handle real landmarks
                else:
                    if landmark_idx < len(pose_landmarks):
                        position = pose_landmarks[landmark_idx]
                        
                        if landmark_idx in pose_rotations:
                            rotation = pose_rotations[landmark_idx]
                            rotation_matrix = rotation.rotation_matrix
                            if hasattr(rotation, 'quaternion'):
                                quaternion = rotation.quaternion
                        else:
                            rotation_matrix = np.eye(3)
                
                # Send if position is available
                if position is not None:
                    self.send_bone_transform(bone_name, position, 
                                           rotation_matrix, quaternion)
            
            # Send root transform (use SACROILIAC if virtual joints enabled, else hip center)
            if self.include_virtual_joints and self.VIRTUAL_SACROILIAC in virtual_positions:
                # Use SACROILIAC as root
                root_position = virtual_positions[self.VIRTUAL_SACROILIAC]
                if self.VIRTUAL_SACROILIAC in pose_rotations:
                    rotation = pose_rotations[self.VIRTUAL_SACROILIAC]
                    if hasattr(rotation, 'quaternion'):
                        self.send_root_transform(root_position, 
                                                quaternion=rotation.quaternion)
                    else:
                        self.send_root_transform(root_position, 
                                                rotation_matrix=rotation.rotation_matrix)
                else:
                    self.send_root_transform(root_position)
            elif len(pose_landmarks) > 24:
                # Fallback to hip center
                hip_center = (pose_landmarks[23] + pose_landmarks[24]) / 2
                self.send_root_transform(hip_center)
        
        # Process left hand landmarks
        if 'left_hand' in landmarks_3d and 'left_hand' in rotations:
            left_hand_landmarks = landmarks_3d['left_hand']
            left_hand_rotations = rotations['left_hand']
            
            for landmark_idx, bone_name in self.VMC_BONE_MAPPING['left_hand'].items():
                if landmark_idx < len(left_hand_landmarks):
                    position = left_hand_landmarks[landmark_idx]
                    
                    if landmark_idx in left_hand_rotations:
                        rotation = left_hand_rotations[landmark_idx]
                        if hasattr(rotation, 'quaternion'):
                            self.send_bone_transform(bone_name, position, 
                                                   quaternion=rotation.quaternion)
                        else:
                            self.send_bone_transform(bone_name, position, 
                                                   rotation_matrix=rotation.rotation_matrix)
                    else:
                        self.send_bone_transform(bone_name, position)
        
        # Process right hand landmarks
        if 'right_hand' in landmarks_3d and 'right_hand' in rotations:
            right_hand_landmarks = landmarks_3d['right_hand']
            right_hand_rotations = rotations['right_hand']
            
            for landmark_idx, bone_name in self.VMC_BONE_MAPPING['right_hand'].items():
                if landmark_idx < len(right_hand_landmarks):
                    position = right_hand_landmarks[landmark_idx]
                    
                    if landmark_idx in right_hand_rotations:
                        rotation = right_hand_rotations[landmark_idx]
                        if hasattr(rotation, 'quaternion'):
                            self.send_bone_transform(bone_name, position, 
                                                   quaternion=rotation.quaternion)
                        else:
                            self.send_bone_transform(bone_name, position, 
                                                   rotation_matrix=rotation.rotation_matrix)
                    else:
                        self.send_bone_transform(bone_name, position)
        
        # Send frame available signal
        self.send_frame_available()


# Example usage and integration
if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("Enhanced OSC/VMC Client - Virtual Joints Support")
    print("="*70)
    print("\nFeatures:")
    print("  ✅ Sends VT1 (Spine) joint data")
    print("  ✅ Sends SACROILIAC (Hips) joint data")
    print("  ✅ Proper spine axis in VR/VMC applications")
    print("  ✅ Compatible with standard VMC protocol")
    print("  ✅ Can be disabled for compatibility")
    print("\nBone Names in VMC:")
    print("  VT1 → 'Spine' (upper spine reference)")
    print("  SACROILIAC → 'Hips' (pelvic center)")
    print("\nThese can be mapped to:")
    print("  - Unity Humanoid rig: Spine/Hips bones")
    print("  - VRM models: Spine/Hips nodes")
    print("  - Custom rigs: Any spine/pelvis reference")
    print("="*70)
    
    # Example initialization
    print("\nExample 1: With virtual joints (default)")
    client1 = EnhancedOSCVMCClient("127.0.0.1", 39539, include_virtual_joints=True)
    
    print("\nExample 2: Without virtual joints (for compatibility)")
    client2 = EnhancedOSCVMCClient("127.0.0.1", 39540, include_virtual_joints=False)
    
    print("\n✅ Ready to integrate with MediaPipe!")
    print("Use client.send_all_bones(landmarks_3d, rotations) to stream data")
