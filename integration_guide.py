"""
Integration Guide: How to Fix Quaternion Issues in Your Code
==============================================================

This guide shows you exactly how to integrate the quaternion fix into your
advanced_examples.py file to resolve the identity quaternion (0,0,0,1) issue
for shoulders and other joints.
"""

# ============================================================================
# STEP 1: Add Import at the Top of Your File
# ============================================================================

# Add this import after your existing imports:
from mediapipe_quaternion_fix import (
    POSE_BONE_HIERARCHY,
    HAND_BONE_HIERARCHY,
    POSE_CONNECTIONS,
    HAND_CONNECTIONS,
    calculate_joint_rotation_from_hierarchy,
    get_all_pose_rotations,
    get_all_hand_rotations,
    print_rotation_info
)

# ============================================================================
# STEP 2: Modify Your MediaPipeJointRotations Class
# ============================================================================

# Option A: Add a new method to calculate rotations using hierarchy
def calculate_rotations_with_hierarchy(self, landmarks_3d_dict):
    """
    Calculate joint rotations using proper bone hierarchies.
    This fixes the identity quaternion issue.
    
    Args:
        landmarks_3d_dict: Dict with 'pose', 'left_hand', 'right_hand' 3D landmarks
        
    Returns:
        Dict with rotations for each joint
    """
    rotations = {}
    
    # Calculate pose joint rotations
    if 'pose' in landmarks_3d_dict and landmarks_3d_dict['pose'] is not None:
        pose_landmarks_dict = {
            i: landmarks_3d_dict['pose'][i] 
            for i in range(len(landmarks_3d_dict['pose']))
        }
        rotations['pose'] = get_all_pose_rotations(pose_landmarks_dict)
    
    # Calculate left hand rotations
    if 'left_hand' in landmarks_3d_dict and landmarks_3d_dict['left_hand'] is not None:
        left_hand_dict = {
            i: landmarks_3d_dict['left_hand'][i]
            for i in range(len(landmarks_3d_dict['left_hand']))
        }
        rotations['left_hand'] = get_all_hand_rotations(left_hand_dict)
    
    # Calculate right hand rotations
    if 'right_hand' in landmarks_3d_dict and landmarks_3d_dict['right_hand'] is not None:
        right_hand_dict = {
            i: landmarks_3d_dict['right_hand'][i]
            for i in range(len(landmarks_3d_dict['right_hand']))
        }
        rotations['right_hand'] = get_all_hand_rotations(right_hand_dict)
    
    return rotations

# ============================================================================
# STEP 3: Example Usage in Your Code
# ============================================================================

def example_test_shoulder_rotations():
    """
    Test example specifically for shoulder rotations.
    This demonstrates that shoulders now have proper quaternions.
    """
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe_quaternion_fix import (
        POSE_BONE_HIERARCHY,
        calculate_joint_rotation_from_hierarchy,
        print_rotation_info
    )
    
    print("\n" + "="*70)
    print("Testing Shoulder Rotation Fix")
    print("="*70)
    
    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        print("\n🎥 Camera started. Move your arms to test shoulder rotations.")
        print("Press 'q' to quit, 's' to print shoulder rotation details\n")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            # Process the image
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )
            
            # Calculate shoulder rotations if pose detected
            if results.pose_world_landmarks:
                # Convert to dictionary format
                landmarks_3d = {
                    i: np.array([lm.x, lm.y, lm.z])
                    for i, lm in enumerate(results.pose_world_landmarks.landmark)
                }
                
                # Calculate left shoulder rotation
                l_parent, l_joint, l_child = POSE_BONE_HIERARCHY['l_shoulder']
                l_shoulder_quat = calculate_joint_rotation_from_hierarchy(
                    landmarks_3d, l_parent, l_joint, l_child
                )
                
                # Calculate right shoulder rotation
                r_parent, r_joint, r_child = POSE_BONE_HIERARCHY['r_shoulder']
                r_shoulder_quat = calculate_joint_rotation_from_hierarchy(
                    landmarks_3d, r_parent, r_joint, r_child
                )
                
                # Display quaternion values on screen
                if l_shoulder_quat is not None:
                    text = f"L Shoulder: [{l_shoulder_quat[0]:.2f}, {l_shoulder_quat[1]:.2f}, {l_shoulder_quat[2]:.2f}, {l_shoulder_quat[3]:.2f}]"
                    cv2.putText(image, text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Check if it's an identity quaternion (the problem we're fixing)
                    if np.allclose(l_shoulder_quat, [0, 0, 0, 1], atol=1e-3):
                        cv2.putText(image, "WARNING: Identity quaternion!", (10, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else:
                        cv2.putText(image, "OK: Valid rotation", (10, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if r_shoulder_quat is not None:
                    text = f"R Shoulder: [{r_shoulder_quat[0]:.2f}, {r_shoulder_quat[1]:.2f}, {r_shoulder_quat[2]:.2f}, {r_shoulder_quat[3]:.2f}]"
                    cv2.putText(image, text, (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow('Shoulder Rotation Test (Press s for details, q to quit)', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Print detailed rotation information
                if results.pose_world_landmarks:
                    print("\n" + "="*70)
                    print("DETAILED SHOULDER ROTATION INFO")
                    print("="*70)
                    if l_shoulder_quat is not None:
                        print_rotation_info("Left Shoulder", l_shoulder_quat)
                    if r_shoulder_quat is not None:
                        print_rotation_info("Right Shoulder", r_shoulder_quat)
                    print("="*70 + "\n")
    
    cap.release()
    cv2.destroyAllWindows()

# ============================================================================
# STEP 4: Update Your OSC/VMC Streaming to Use Fixed Quaternions
# ============================================================================

def send_fixed_quaternions_via_osc(osc_client, rotations):
    """
    Send the properly calculated quaternions via OSC.
    Use this instead of the old method that produced identity quaternions.
    """
    # Send pose rotations
    if 'pose' in rotations:
        for joint_name, quaternion in rotations['pose'].items():
            # Map joint name to VMC bone name
            vmc_bone_name = f"/VMC/Ext/Bone/Pos/{joint_name}"
            
            # Send quaternion (x, y, z, w)
            osc_client.send_message(vmc_bone_name, [
                quaternion[0],  # x
                quaternion[1],  # y
                quaternion[2],  # z
                quaternion[3]   # w
            ])
    
    # Send hand rotations
    for hand_side in ['left_hand', 'right_hand']:
        if hand_side in rotations:
            for joint_name, quaternion in rotations[hand_side].items():
                vmc_bone_name = f"/VMC/Ext/Bone/Pos/{hand_side}_{joint_name}"
                osc_client.send_message(vmc_bone_name, [
                    quaternion[0], quaternion[1], quaternion[2], quaternion[3]
                ])

# ============================================================================
# COMPLETE EXAMPLE: Modified AdvancedJointAnalyzer Class
# ============================================================================

class FixedAdvancedJointAnalyzer:
    """
    Example of how to integrate the quaternion fix into your existing class.
    """
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def process_frame_with_fixed_rotations(self, frame):
        """
        Process a frame and return fixed quaternion rotations.
        """
        # Convert image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process with MediaPipe
        results = self.holistic.process(image)
        
        # Convert back for drawing
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS
            )
        
        # Calculate rotations using the FIX
        rotations = {}
        
        if results.pose_world_landmarks:
            pose_3d = {
                i: np.array([lm.x, lm.y, lm.z])
                for i, lm in enumerate(results.pose_world_landmarks.landmark)
            }
            rotations['pose'] = get_all_pose_rotations(pose_3d)
        
        if results.left_hand_world_landmarks:
            left_hand_3d = {
                i: np.array([lm.x, lm.y, lm.z])
                for i, lm in enumerate(results.left_hand_world_landmarks.landmark)
            }
            rotations['left_hand'] = get_all_hand_rotations(left_hand_3d)
        
        if results.right_hand_world_landmarks:
            right_hand_3d = {
                i: np.array([lm.x, lm.y, lm.z])
                for i, lm in enumerate(results.right_hand_world_landmarks.landmark)
            }
            rotations['right_hand'] = get_all_hand_rotations(right_hand_3d)
        
        return image, rotations

# ============================================================================
# RUN THE TEST
# ============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("Choose an option:")
    print("  1. Test shoulder rotations (recommended)")
    print("  2. Show integration code")
    print("="*70)
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        example_test_shoulder_rotations()
    else:
        print("\n✅ Review the code above to see how to integrate the fix!")
        print("📝 Key points:")
        print("   1. Import the fix module")
        print("   2. Use POSE_BONE_HIERARCHY and HAND_BONE_HIERARCHY")
        print("   3. Call get_all_pose_rotations() and get_all_hand_rotations()")
        print("   4. You'll get proper quaternions instead of (0,0,0,1)!")
