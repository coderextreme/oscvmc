"""
Standalone OSC/VMC Protocol Streaming Client
=============================================
Streams MediaPipe Holistic tracking data to VMC-compatible applications
via the Open Sound Control (OSC) protocol.

Compatible with: VSeeFace, VMagicMirror, Virtual Motion Capture, and other VMC apps
"""

import cv2
import numpy as np
import time
import argparse
from pythonosc import udp_client
from mediapipe_joint_rotations import MediaPipeJointRotations
import mediapipe as mp

mp_hands = mp.solutions.hands.HandLandmark
poselm = mp.solutions.pose.PoseLandmark

hand = [[mp_hands.WRIST, "WRIST", "radiocarpal"],
[mp_hands.THUMB_CMC, "THUMB_CMC", "carpometacarpal_1"],
[mp_hands.THUMB_MCP, "THUMB_MCP", "metacarpophalangeal_1"],
[mp_hands.THUMB_IP, "THUMB_IP", "carpal_interphalangeal_1"],
[mp_hands.THUMB_TIP, "THUMB_TIP", "carpal_distal_phalanx_1"],
[mp_hands.INDEX_FINGER_MCP, "INDEX_FINGER_MCP", "metacarpophalangeal_2"],
[mp_hands.INDEX_FINGER_PIP, "INDEX_FINGER_PIP", "carpal_proximal_interphalangeal_2"],
[mp_hands.INDEX_FINGER_DIP, "INDEX_FINGER_DIP", "carpal_distal_interphalangeal_2"],
[mp_hands.INDEX_FINGER_TIP, "INDEX_FINGER_TIP", "carpal_distal_phalanx_2"],
[mp_hands.MIDDLE_FINGER_MCP, "MIDDLE_FINGER_MCP", "metacarpophalangeal_3"],
[mp_hands.MIDDLE_FINGER_PIP, "MIDDLE_FINGER_PIP", "carpal_proximal_interphalangeal_3"],
[mp_hands.MIDDLE_FINGER_DIP, "MIDDLE_FINGER_DIP", "carpal_distal_interphalangeal_3"],
[mp_hands.MIDDLE_FINGER_TIP, "MIDDLE_FINGER_TIP", "carpal_distal_phalanx_3"],
[mp_hands.RING_FINGER_MCP, "RING_FINGER_MCP", "metacarpophalangeal_4"],
[mp_hands.RING_FINGER_PIP, "RING_FINGER_PIP", "carpal_proximal_interphalangeal_4"],
[mp_hands.RING_FINGER_DIP, "RING_FINGER_DIP", "carpal_distal_interphalangeal_4"],
[mp_hands.RING_FINGER_TIP, "RING_FINGER_TIP", "carpal_distal_phalanx_4"],
[mp_hands.PINKY_MCP, "PINKY_MCP", "metacarpophalangeal_5"],
[mp_hands.PINKY_PIP, "PINKY_PIP", "carpal_proximal_interphalangeal_5"],
[mp_hands.PINKY_DIP, "PINKY_DIP", "carpal_distal_interphalangeal_5"],
[mp_hands.PINKY_TIP, "PINKY_TIP", "carpal_distal_phalanx_5"]]

handmap = {}

for h in hand:
    handmap[h[0]] = h[2]

pose = [
[poselm.NOSE, 0, "nose"],
[poselm.LEFT_EYE_INNER, 1, "l_eye_inner"],
[poselm.LEFT_EYE, 2, "l_eye"],
[poselm.LEFT_EYE_OUTER, 3, "l_eye_outer"],
[poselm.RIGHT_EYE_INNER, 4, "r_eye_inner"],
[poselm.RIGHT_EYE, 5, "r_eye"],
[poselm.RIGHT_EYE_OUTER, 6, "r_eye_outer"],
[poselm.LEFT_EAR, 7, "l_ear"],
[poselm.RIGHT_EAR, 8, "r_ear"],
[poselm.MOUTH_LEFT, 9, "l_mouth"],
[poselm.MOUTH_RIGHT, 10, "r_mouth"],
[poselm.LEFT_SHOULDER, 11, "l_shoulder"],
[poselm.RIGHT_SHOULDER, 12, "r_shoulder"],
[poselm.LEFT_ELBOW, 13, "l_elbow"],
[poselm.RIGHT_ELBOW, 14, "r_elbow"],
[poselm.LEFT_WRIST, 15, "l_wrist"],
[poselm.RIGHT_WRIST, 16, "r_wrist"],
[poselm.LEFT_PINKY, 17, "l_pinky"],
[poselm.RIGHT_PINKY, 18, "r_pinky"],
[poselm.LEFT_INDEX, 19, "l_index"],
[poselm.RIGHT_INDEX, 20, "r_index"],
[poselm.LEFT_THUMB, 21, "l_thumb"],
[poselm.RIGHT_THUMB, 22, "r_thumb"],
[poselm.LEFT_HIP, 23, "l_hip"],
[poselm.RIGHT_HIP, 24, "r_hip"],
[poselm.LEFT_KNEE, 25, "l_knee"],
[poselm.RIGHT_KNEE, 26, "r_knee"],
[poselm.LEFT_ANKLE, 27, "l_talocrural"],
[poselm.RIGHT_ANKLE, 28, "r_talocrural"],
[poselm.LEFT_HEEL, 29, "l_heel"],
[poselm.RIGHT_HEEL, 30, "r_heel"],
[poselm.LEFT_FOOT_INDEX, 31, "l_metatarsophalangeal_2"],
[poselm.RIGHT_FOOT_INDEX, 32, "r_metatarsophalangeal_2"]
]

posemap = {}

for p in pose:
    posemap[p[1]] = p[2]

class VMCProtocolStreamer:
    """
    Complete VMC Protocol implementation for motion capture streaming
    """
    
    # Complete VMC bone mapping
    BONE_MAPPING = {
        # Spine and core
        'Hips': 'root',
        'Spine': 'spine',
        'Chest': 'chest',
        'Neck': 'neck',
        'Head': 'head',
        
        # Left arm chain
        'LeftShoulder': 11,
        'LeftUpperArm': 11,
        'LeftLowerArm': 13,
        'LeftHand': 15,
        
        # Right arm chain  
        'RightShoulder': 12,
        'RightUpperArm': 12,
        'RightLowerArm': 14,
        'RightHand': 16,
        
        # Left leg chain
        'LeftUpperLeg': 23,
        'LeftLowerLeg': 25,
        'LeftFoot': 27,
        'LeftToes': 31,
        
        # Right leg chain
        'RightUpperLeg': 24,
        'RightLowerLeg': 26,
        'RightFoot': 28,
        'RightToes': 32,
    }
    
    # Hand bone mapping (finger joints)
    HAND_BONES = {
        'left': {
            'LeftThumbProximal': 1,
            'LeftThumbIntermediate': 2,
            'LeftThumbDistal': 3,
            'LeftIndexProximal': 5,
            'LeftIndexIntermediate': 6,
            'LeftIndexDistal': 7,
            'LeftMiddleProximal': 9,
            'LeftMiddleIntermediate': 10,
            'LeftMiddleDistal': 11,
            'LeftRingProximal': 13,
            'LeftRingIntermediate': 14,
            'LeftRingDistal': 15,
            'LeftLittleProximal': 17,
            'LeftLittleIntermediate': 18,
            'LeftLittleDistal': 19,
        },
        'right': {
            'RightThumbProximal': 1,
            'RightThumbIntermediate': 2,
            'RightThumbDistal': 3,
            'RightIndexProximal': 5,
            'RightIndexIntermediate': 6,
            'RightIndexDistal': 7,
            'RightMiddleProximal': 9,
            'RightMiddleIntermediate': 10,
            'RightMiddleDistal': 11,
            'RightRingProximal': 13,
            'RightRingIntermediate': 14,
            'RightRingDistal': 15,
            'RightLittleProximal': 17,
            'RightLittleIntermediate': 18,
            'RightLittleDistal': 19,
        }
    }
    
    def __init__(self, ip: str = "127.0.0.1", port: int = 39539, 
                 verbose: bool = False):
        """
        Initialize VMC Protocol Streamer
        
        Args:
            ip: Target IP address
            port: Target port (39539 is VMC protocol standard)
            verbose: Print detailed OSC messages
        """
        self.ip = ip
        self.port = port
        self.verbose = verbose
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.tracker = MediaPipeJointRotations()
        self.frame_count = 0
        self.start_time = time.time()
        
        print(f"╔{'═'*68}╗")
        print(f"║{'VMC Protocol Streamer Initialized':^68}║")
        print(f"╠{'═'*68}╣")
        print(f"║ Target: {ip}:{port:<57}║")
        print(f"║ Protocol: VMC (Virtual Motion Capture){' '*28}║")
        print(f"║ Port 39539: Standard VMC receiver port{' '*29}║")
        print(f"╚{'═'*68}╝")
    
    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> tuple:
        """
        Convert 3x3 rotation matrix to quaternion (x, y, z, w)
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Tuple of (qx, qy, qz, qw)
        """
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
        
        return (x, y, z, w)
    
    def send_bone_position(self, bone_name: str, pos: np.ndarray, 
                          rot_matrix: np.ndarray):
        """
        Send bone transform via VMC protocol
        
        VMC Message Format:
        /VMC/Ext/Bone/Pos {name} {px} {py} {pz} {qx} {qy} {qz} {qw}
        
        Args:
            bone_name: VMC bone name
            pos: 3D position [x, y, z]
            rot_matrix: 3x3 rotation matrix
        """
        qx, qy, qz, qw = self.rotation_matrix_to_quaternion(rot_matrix)
        
        message = [
            bone_name,
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(qx),
            float(qy),
            float(qz),
            float(qw)
        ]
        
        self.client.send_message("/VMC/Ext/Bone/Pos", message)
        
        if self.verbose:
            print(f"  → {bone_name}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                  f"quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")
    
    def send_root_position(self, pos: np.ndarray, rot_matrix: np.ndarray):
        """
        Send root transform (Hips)
        
        VMC Message Format:
        /VMC/Ext/Root/Pos {name} {px} {py} {pz} {qx} {qy} {qz} {qw} {sx} {sy} {sz} {o}
        """
        qx, qy, qz, qw = self.rotation_matrix_to_quaternion(rot_matrix)
        
        message = [
            "root",
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(qx),
            float(qy),
            float(qz),
            float(qw),
            0.0, 0.0, 0.0,  # Scale (unused)
            0.0  # Offset (unused)
        ]
        
        self.client.send_message("/VMC/Ext/Root/Pos", message)
    
    def send_frame_time(self):
        """Send current time"""
        elapsed = time.time() - self.start_time
        self.client.send_message("/VMC/Ext/T", [float(elapsed)])
    
    def send_ok(self):
        """Signal frame is complete and ready"""
        self.client.send_message("/VMC/Ext/OK", [1])
        self.frame_count += 1
    
    def process_and_stream_frame(self, frame: np.ndarray) -> tuple:
        """
        Process frame and stream all bone data via VMC protocol
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (annotated_frame, bone_count)
        """
        # Process with MediaPipe
        annotated_frame, all_rotations = self.tracker.process_frame(frame)
        
        # Get raw results for 3D positions
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.tracker.holistic.process(image_rgb)
        
        bone_count = 0
        
        # Send frame time
        self.send_frame_time()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Frame {self.frame_count}")
            print(f"{'='*70}")
        
        # Process pose landmarks
        if results.pose_landmarks:
            pose_landmarks = self.tracker.extract_3d_landmarks(results.pose_landmarks)
            pose_rotations = all_rotations.get('pose', {})
            
            # Send root (hip center)
            if len(pose_landmarks) > 24:
                hip_center = (pose_landmarks[23] + pose_landmarks[24]) / 2
                root_rotation = pose_rotations.get(23, None)
                rot_matrix = root_rotation.rotation_matrix if root_rotation else np.eye(3)
                self.send_root_position(hip_center, rot_matrix)
                bone_count += 1
            
            # Send pose bones
            for bone_name, landmark_idx in self.BONE_MAPPING.items():
                if isinstance(landmark_idx, int) and landmark_idx < len(pose_landmarks):
                    position = pose_landmarks[landmark_idx]
                    rotation = pose_rotations.get(landmark_idx)
                    rot_matrix = rotation.rotation_matrix if rotation else np.eye(3)
                    
                    self.send_bone_position(posemap[landmark_idx], position, rot_matrix)
                    # self.send_bone_position(bone_name, position, rot_matrix)
                    bone_count += 1
        
        # Process left hand
        if results.left_hand_landmarks:
            left_hand = self.tracker.extract_3d_landmarks(results.left_hand_landmarks)
            left_rotations = all_rotations.get('left_hand', {})
            
            for bone_name, landmark_idx in self.HAND_BONES['left'].items():
                if landmark_idx < len(left_hand):
                    position = left_hand[landmark_idx]
                    rotation = left_rotations.get(landmark_idx)
                    rot_matrix = rotation.rotation_matrix if rotation else np.eye(3)
                    
                    # self.send_bone_position(bone_name, position, rot_matrix)
                    self.send_bone_position("l_"+handmap[landmark_idx], position, rot_matrix)
                    bone_count += 1
        
        # Process right hand
        if results.right_hand_landmarks:
            right_hand = self.tracker.extract_3d_landmarks(results.right_hand_landmarks)
            right_rotations = all_rotations.get('right_hand', {})
            
            for bone_name, landmark_idx in self.HAND_BONES['right'].items():
                if landmark_idx < len(right_hand):
                    position = right_hand[landmark_idx]
                    rotation = right_rotations.get(landmark_idx)
                    rot_matrix = rotation.rotation_matrix if rotation else np.eye(3)
                    
                    # self.send_bone_position(bone_name, position, rot_matrix)
                    self.send_bone_position("r_"+handmap[landmark_idx], position, rot_matrix)
                    bone_count += 1
        
        # Signal frame complete
        self.send_ok()
        
        return annotated_frame, bone_count


def main():
    """Main streaming loop"""
    parser = argparse.ArgumentParser(
        description='Stream MediaPipe tracking to VMC protocol server'
    )
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                       help='OSC server IP address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=39539,
                       help='OSC server port (default: 39539)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed OSC messages')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display window')
    
    args = parser.parse_args()
    
    # Initialize streamer
    streamer = VMCProtocolStreamer(args.ip, args.port, args.verbose)
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        return
    
    print("\n" + "="*70)
    print("STREAMING CONTROLS:")
    print("  Q - Quit")
    print("  P - Pause/Resume streaming")
    print("  V - Toggle verbose output")
    print("  D - Toggle display window")
    print("="*70 + "\n")
    
    # Streaming state
    is_streaming = True
    show_display = not args.no_display
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                continue
            
            # Process and stream
            if is_streaming:
                annotated_frame, bone_count = streamer.process_and_stream_frame(frame)
            else:
                annotated_frame = frame
                bone_count = 0
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()
            
            # Display window
            if show_display:
                # Add status overlay
                status_text = "STREAMING" if is_streaming else "PAUSED"
                status_color = (0, 255, 0) if is_streaming else (0, 0, 255)
                
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                           (annotated_frame.shape[1] - 120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(annotated_frame, f"VMC: {status_text}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                cv2.putText(annotated_frame, f"Server: {args.ip}:{args.port}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(annotated_frame, f"Bones: {bone_count}",
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(annotated_frame, f"Frames: {streamer.frame_count}",
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('VMC Protocol Streamer', annotated_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                is_streaming = not is_streaming
                print(f"→ Streaming {'RESUMED' if is_streaming else 'PAUSED'}")
            elif key == ord('v'):
                streamer.verbose = not streamer.verbose
                print(f"→ Verbose mode {'ON' if streamer.verbose else 'OFF'}")
            elif key == ord('d'):
                show_display = not show_display
                if not show_display:
                    cv2.destroyAllWindows()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        print(f"Session Summary:")
        print(f"  Total frames streamed: {streamer.frame_count}")
        print(f"  Duration: {time.time() - streamer.start_time:.1f}s")
        print(f"  Average FPS: {streamer.frame_count / (time.time() - streamer.start_time):.1f}")
        print("="*70)


if __name__ == "__main__":
    main()
