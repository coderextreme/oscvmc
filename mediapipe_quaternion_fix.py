"""
MediaPipe Quaternion Fix - Enhanced Joint Rotation Calculator
==============================================================

This module provides the necessary additions to fix quaternion calculation
issues in MediaPipe joint rotation tracking, specifically for shoulders and
other joints that were returning identity quaternions (0,0,0,1).

Key improvements:
1. Proper bone hierarchies (parent-joint-child relationships)
2. Official MediaPipe POSE_CONNECTIONS and HAND_CONNECTIONS
3. Enhanced quaternion calculation using proper vector relationships
4. Support for all pose and hand joints

Usage:
    Add this import to your advanced_examples.py:
    from mediapipe_quaternion_fix import *
    
    Then use POSE_BONE_HIERARCHY and HAND_BONE_HIERARCHY in your
    MediaPipeJointRotations class.
"""

import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

# MediaPipe landmark enums
mp_pose = mp.solutions.pose.PoseLandmark
mp_hands = mp.solutions.hands.HandLandmark

# ============================================================================
# MEDIAPIPE OFFICIAL CONNECTIONS
# ============================================================================

# These are the official MediaPipe connections - DO NOT MODIFY
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

print(f"Loaded {len(POSE_CONNECTIONS)} pose connections")
print(f"Loaded {len(HAND_CONNECTIONS)} hand connections")

# ============================================================================
# BONE HIERARCHIES FOR QUATERNION CALCULATION
# ============================================================================

# Each entry is (parent_landmark, joint_landmark, child_landmark)
# The quaternion represents the rotation at 'joint_landmark' from the
# parent-to-joint vector to the joint-to-child vector

POSE_BONE_HIERARCHY = {
    # ========== UPPER BODY ==========
    
    # Neck/Head (using shoulder midpoint as reference)
    'neck': (mp_pose.LEFT_SHOULDER, mp_pose.RIGHT_SHOULDER, mp_pose.NOSE),
    
    # Left arm chain
    'l_shoulder': (mp_pose.LEFT_HIP, mp_pose.LEFT_SHOULDER, mp_pose.LEFT_ELBOW),
    'l_elbow': (mp_pose.LEFT_SHOULDER, mp_pose.LEFT_ELBOW, mp_pose.LEFT_WRIST),
    'l_wrist': (mp_pose.LEFT_ELBOW, mp_pose.LEFT_WRIST, mp_pose.LEFT_INDEX),
    
    # Right arm chain
    'r_shoulder': (mp_pose.RIGHT_HIP, mp_pose.RIGHT_SHOULDER, mp_pose.RIGHT_ELBOW),
    'r_elbow': (mp_pose.RIGHT_SHOULDER, mp_pose.RIGHT_ELBOW, mp_pose.RIGHT_WRIST),
    'r_wrist': (mp_pose.RIGHT_ELBOW, mp_pose.RIGHT_WRIST, mp_pose.RIGHT_INDEX),
    
    # ========== LOWER BODY ==========
    
    # Left leg chain
    'l_hip': (mp_pose.LEFT_SHOULDER, mp_pose.LEFT_HIP, mp_pose.LEFT_KNEE),
    'l_knee': (mp_pose.LEFT_HIP, mp_pose.LEFT_KNEE, mp_pose.LEFT_ANKLE),
    'l_ankle': (mp_pose.LEFT_KNEE, mp_pose.LEFT_ANKLE, mp_pose.LEFT_FOOT_INDEX),
    'l_foot': (mp_pose.LEFT_ANKLE, mp_pose.LEFT_FOOT_INDEX, mp_pose.LEFT_HEEL),
    
    # Right leg chain
    'r_hip': (mp_pose.RIGHT_SHOULDER, mp_pose.RIGHT_HIP, mp_pose.RIGHT_KNEE),
    'r_knee': (mp_pose.RIGHT_HIP, mp_pose.RIGHT_KNEE, mp_pose.RIGHT_ANKLE),
    'r_ankle': (mp_pose.RIGHT_KNEE, mp_pose.RIGHT_ANKLE, mp_pose.RIGHT_FOOT_INDEX),
    'r_foot': (mp_pose.RIGHT_ANKLE, mp_pose.RIGHT_FOOT_INDEX, mp_pose.RIGHT_HEEL),
}

HAND_BONE_HIERARCHY = {
    # ========== THUMB ==========
    'thumb_cmc': (mp_hands.WRIST, mp_hands.THUMB_CMC, mp_hands.THUMB_MCP),
    'thumb_mcp': (mp_hands.THUMB_CMC, mp_hands.THUMB_MCP, mp_hands.THUMB_IP),
    'thumb_ip': (mp_hands.THUMB_MCP, mp_hands.THUMB_IP, mp_hands.THUMB_TIP),
    
    # ========== INDEX FINGER ==========
    'index_mcp': (mp_hands.WRIST, mp_hands.INDEX_FINGER_MCP, mp_hands.INDEX_FINGER_PIP),
    'index_pip': (mp_hands.INDEX_FINGER_MCP, mp_hands.INDEX_FINGER_PIP, mp_hands.INDEX_FINGER_DIP),
    'index_dip': (mp_hands.INDEX_FINGER_PIP, mp_hands.INDEX_FINGER_DIP, mp_hands.INDEX_FINGER_TIP),
    
    # ========== MIDDLE FINGER ==========
    'middle_mcp': (mp_hands.WRIST, mp_hands.MIDDLE_FINGER_MCP, mp_hands.MIDDLE_FINGER_PIP),
    'middle_pip': (mp_hands.MIDDLE_FINGER_MCP, mp_hands.MIDDLE_FINGER_PIP, mp_hands.MIDDLE_FINGER_DIP),
    'middle_dip': (mp_hands.MIDDLE_FINGER_PIP, mp_hands.MIDDLE_FINGER_DIP, mp_hands.MIDDLE_FINGER_TIP),
    
    # ========== RING FINGER ==========
    'ring_mcp': (mp_hands.WRIST, mp_hands.RING_FINGER_MCP, mp_hands.RING_FINGER_PIP),
    'ring_pip': (mp_hands.RING_FINGER_MCP, mp_hands.RING_FINGER_PIP, mp_hands.RING_FINGER_DIP),
    'ring_dip': (mp_hands.RING_FINGER_PIP, mp_hands.RING_FINGER_DIP, mp_hands.RING_FINGER_TIP),
    
    # ========== PINKY FINGER ==========
    'pinky_mcp': (mp_hands.WRIST, mp_hands.PINKY_MCP, mp_hands.PINKY_PIP),
    'pinky_pip': (mp_hands.PINKY_MCP, mp_hands.PINKY_PIP, mp_hands.PINKY_DIP),
    'pinky_dip': (mp_hands.PINKY_PIP, mp_hands.PINKY_DIP, mp_hands.PINKY_TIP),
    
    # ========== WRIST ==========
    # Wrist uses middle finger MCP and thumb CMC to define its orientation
    'wrist': (mp_hands.MIDDLE_FINGER_MCP, mp_hands.WRIST, mp_hands.THUMB_CMC),
}

# ============================================================================
# ENHANCED QUATERNION CALCULATION FUNCTIONS
# ============================================================================

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length"""
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])  # Default to z-axis if zero vector
    return v / norm


def quaternion_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """
    Calculate quaternion that rotates from v_from to v_to.
    
    This is the KEY function for fixing the shoulder quaternion issue.
    
    Args:
        v_from: Source vector (e.g., parent-to-joint direction)
        v_to: Target vector (e.g., joint-to-child direction)
        
    Returns:
        Quaternion as [x, y, z, w]
    """
    # Normalize input vectors
    v1 = normalize_vector(v_from)
    v2 = normalize_vector(v_to)
    
    # Calculate the cross product (rotation axis)
    axis = np.cross(v1, v2)
    axis_length = np.linalg.norm(axis)
    
    # Calculate dot product (related to rotation angle)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    
    # Handle special cases
    if axis_length < 1e-6:
        # Vectors are parallel or anti-parallel
        if dot > 0:
            # Same direction - no rotation needed
            return np.array([0.0, 0.0, 0.0, 1.0])
        else:
            # Opposite direction - 180 degree rotation
            # Find a perpendicular axis
            if abs(v1[0]) < 0.9:
                axis = np.cross(v1, np.array([1, 0, 0]))
            else:
                axis = np.cross(v1, np.array([0, 1, 0]))
            axis = normalize_vector(axis)
            return np.array([axis[0], axis[1], axis[2], 0.0])
    
    # Normalize the axis
    axis = axis / axis_length
    
    # Calculate quaternion components using half-angle formulas
    # q = [sin(θ/2) * axis, cos(θ/2)]
    half_angle = np.arccos(dot) / 2.0
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)
    
    quat = np.array([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        cos_half
    ])
    
    return quat


def calculate_joint_rotation_from_hierarchy(
    landmarks_3d: Dict[int, np.ndarray],
    parent_idx: int,
    joint_idx: int,
    child_idx: int
) -> Optional[np.ndarray]:
    """
    Calculate joint rotation quaternion using bone hierarchy.
    
    This is the PROPER way to calculate joint rotations that fixes
    the identity quaternion issue.
    
    Args:
        landmarks_3d: Dictionary mapping landmark index to 3D position
        parent_idx: Index of parent landmark
        joint_idx: Index of joint landmark
        child_idx: Index of child landmark
        
    Returns:
        Quaternion [x, y, z, w] or None if landmarks missing
    """
    # Check if all landmarks are present
    if (parent_idx not in landmarks_3d or
        joint_idx not in landmarks_3d or
        child_idx not in landmarks_3d):
        return None
    
    # Get 3D positions
    parent_pos = landmarks_3d[parent_idx]
    joint_pos = landmarks_3d[joint_idx]
    child_pos = landmarks_3d[child_idx]
    
    # Calculate bone vectors
    # v1: from joint TO parent (inbound)
    # v2: from joint TO child (outbound)
    v1 = parent_pos - joint_pos
    v2 = child_pos - joint_pos
    
    # Calculate rotation from v1 to v2
    quaternion = quaternion_from_two_vectors(v1, v2)
    
    return quaternion


def get_all_pose_rotations(landmarks_3d: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate all pose joint rotations using the bone hierarchy.
    
    Args:
        landmarks_3d: Dictionary mapping pose landmark index to 3D position
        
    Returns:
        Dictionary mapping joint name to quaternion
    """
    rotations = {}
    
    for joint_name, (parent_idx, joint_idx, child_idx) in POSE_BONE_HIERARCHY.items():
        quat = calculate_joint_rotation_from_hierarchy(
            landmarks_3d, parent_idx, joint_idx, child_idx
        )
        if quat is not None:
            rotations[joint_name] = quat
    
    return rotations


def get_all_hand_rotations(landmarks_3d: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate all hand joint rotations using the bone hierarchy.
    
    Args:
        landmarks_3d: Dictionary mapping hand landmark index to 3D position
        
    Returns:
        Dictionary mapping joint name to quaternion
    """
    rotations = {}
    
    for joint_name, (parent_idx, joint_idx, child_idx) in HAND_BONE_HIERARCHY.items():
        quat = calculate_joint_rotation_from_hierarchy(
            landmarks_3d, parent_idx, joint_idx, child_idx
        )
        if quat is not None:
            rotations[joint_name] = quat
    
    return rotations


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in radians.
    
    Args:
        quat: Quaternion as [x, y, z, w]
        
    Returns:
        Euler angles as [roll, pitch, yaw] in radians
    """
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def print_rotation_info(joint_name: str, quaternion: np.ndarray):
    """Print detailed information about a joint rotation"""
    euler = quaternion_to_euler(quaternion)
    euler_deg = np.degrees(euler)
    
    print(f"\n{joint_name}:")
    print(f"  Quaternion: [{quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f}]")
    print(f"  Euler (deg): Roll={euler_deg[0]:.2f}°, Pitch={euler_deg[1]:.2f}°, Yaw={euler_deg[2]:.2f}°")
    
    # Check if it's an identity quaternion (the problem case)
    if np.allclose(quaternion, [0, 0, 0, 1], atol=1e-3):
        print(f"  ⚠️  WARNING: Identity quaternion detected (no rotation)")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MediaPipe Quaternion Fix - Connection and Hierarchy Summary")
    print("="*70)
    
    print(f"\n📊 POSE Connections: {len(POSE_CONNECTIONS)} pairs")
    print(f"📊 HAND Connections: {len(HAND_CONNECTIONS)} pairs")
    
    print(f"\n🦴 POSE Bone Hierarchy: {len(POSE_BONE_HIERARCHY)} joints")
    print("\nKey joints for quaternion calculation:")
    for joint_name in ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow']:
        if joint_name in POSE_BONE_HIERARCHY:
            parent, joint, child = POSE_BONE_HIERARCHY[joint_name]
            print(f"  {joint_name:15s}: parent={parent:2d} → joint={joint:2d} → child={child:2d}")
    
    print(f"\n🖐️  HAND Bone Hierarchy: {len(HAND_BONE_HIERARCHY)} joints")
    print("\nSample hand joints:")
    for joint_name in ['wrist', 'thumb_mcp', 'index_mcp', 'middle_mcp']:
        if joint_name in HAND_BONE_HIERARCHY:
            parent, joint, child = HAND_BONE_HIERARCHY[joint_name]
            print(f"  {joint_name:15s}: parent={parent:2d} → joint={joint:2d} → child={child:2d}")
    
    print("\n" + "="*70)
    print("✅ Ready to use! Import this module in your code.")
    print("="*70)
