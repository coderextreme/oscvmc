"""
Complete Integration Example: MediaPipe ENHANCED + OSC/VMC with Virtual Joints
==============================================================================

This example shows how to use the ENHANCED MediaPipe version (with VT1 and SACROILIAC)
together with the Enhanced OSC/VMC client to stream full body tracking with spine data.

Usage:
    python enhanced_mediapipe_osc_integration.py

Press 'q' to quit, 'p' to pause/resume streaming
"""

import cv2
import numpy as np
from enhanced_osc_vmc_client import EnhancedOSCVMCClient
from mediapipe_joint_rotations_ENHANCED import MediaPipeJointRotations
import time


class MediaPipeOSCStreamer:
    """
    Complete MediaPipe to OSC/VMC streamer with virtual joints support
    """
    
    def __init__(self, osc_ip: str = "127.0.0.1", osc_port: int = 39539,
                 include_virtual_joints: bool = True):
        """
        Initialize streamer
        
        Args:
            osc_ip: OSC server IP
            osc_port: OSC server port (default: 39539 - VMC standard)
            include_virtual_joints: Include VT1 and SACROILIAC (default: True)
        """
        print("Initializing MediaPipe OSC Streamer...")
        print("="*70)
        
        # Initialize MediaPipe with ENHANCED version (has virtual joints)
        self.mp_tracker = MediaPipeJointRotations()
        
        # Initialize Enhanced OSC/VMC client
        self.osc_client = EnhancedOSCVMCClient(
            ip=osc_ip, 
            port=osc_port,
            include_virtual_joints=include_virtual_joints
        )
        
        self.streaming_enabled = True
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0
        
        print("="*70)
        print("✅ Initialization complete!")
        print(f"📡 Streaming to {osc_ip}:{osc_port}")
        if include_virtual_joints:
            print("✨ Virtual joints enabled: VT1 (Spine) + SACROILIAC (Hips)")
        print("="*70)
    
    def prepare_landmarks_for_osc(self, results) -> dict:
        """
        Convert MediaPipe results to format needed by OSC client
        
        Args:
            results: MediaPipe holistic results
            
        Returns:
            Dictionary with 'pose', 'left_hand', 'right_hand' landmark arrays
        """
        landmarks_3d = {}
        
        # Extract pose landmarks
        if results.pose_world_landmarks:
            landmarks_3d['pose'] = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.pose_world_landmarks.landmark
            ])
        
        # Extract left hand landmarks
        if results.left_hand_world_landmarks:
            landmarks_3d['left_hand'] = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.left_hand_world_landmarks.landmark
            ])
        
        # Extract right hand landmarks
        if results.right_hand_world_landmarks:
            landmarks_3d['right_hand'] = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.right_hand_world_landmarks.landmark
            ])
        
        return landmarks_3d
    
    def draw_streaming_info(self, frame: np.ndarray, rotations: dict):
        """
        Draw streaming status and info on frame
        
        Args:
            frame: Video frame
            rotations: Joint rotations dict
        """
        h, w = frame.shape[:2]
        
        # Streaming status
        y_pos = h - 120
        status_color = (0, 255, 0) if self.streaming_enabled else (0, 0, 255)
        status_text = "STREAMING" if self.streaming_enabled else "PAUSED"
        
        cv2.rectangle(frame, (10, y_pos - 25), (250, h - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, y_pos - 25), (250, h - 10), status_color, 2)
        
        cv2.putText(frame, f"OSC/VMC: {status_text}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        y_pos += 25
        
        cv2.putText(frame, f"Server: {self.osc_client.ip}:{self.osc_client.port}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        
        cv2.putText(frame, f"Frames: {self.osc_client.frame_count}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Count active bones
        bone_count = 0
        if 'pose' in rotations:
            bone_count += len(rotations['pose'])
        if 'left_hand' in rotations:
            bone_count += len(rotations['left_hand'])
        if 'right_hand' in rotations:
            bone_count += len(rotations['right_hand'])
        
        y_pos += 20
        cv2.putText(frame, f"Bones: {bone_count}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Virtual joints indicator
        if self.osc_client.include_virtual_joints:
            y_pos = 30
            cv2.putText(frame, "Virtual Joints Active:", 
                       (w - 250, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
            
            # Check if VT1 is in rotations
            if 'pose' in rotations and -1 in rotations['pose']:
                cv2.putText(frame, "VT1 (Spine)", 
                           (w - 250, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                y_pos += 18
            
            # Check if SACROILIAC is in rotations
            if 'pose' in rotations and -2 in rotations['pose']:
                cv2.putText(frame, "SACRO (Hips)", 
                           (w - 250, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                   (w - 120, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Controls
        cv2.putText(frame, "Press 'p' to pause/resume, 'q' to quit", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def process_and_stream(self, frame: np.ndarray):
        """
        Process frame with MediaPipe and stream via OSC/VMC
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with visualizations
        """
        # Process with MediaPipe (ENHANCED version with virtual joints)
        annotated_frame, rotations = self.mp_tracker.process_frame(frame)
        
        # Prepare landmarks for OSC
        # We need to get the raw MediaPipe results for 3D world landmarks
        # Since process_frame doesn't return them, we need to access them differently
        # For this example, we'll extract from the mp_tracker's last results
        
        # Convert to format needed by OSC client
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_tracker.holistic.process(image_rgb)
        
        if self.streaming_enabled:
            landmarks_3d = self.prepare_landmarks_for_osc(results)
            
            # Stream via OSC/VMC (includes virtual joints automatically!)
            self.osc_client.send_all_bones(landmarks_3d, rotations)
        
        # Update FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.fps = 30 / (time.time() - self.fps_start_time)
            self.fps_start_time = time.time()
        
        # Draw streaming info
        self.draw_streaming_info(annotated_frame, rotations)
        
        return annotated_frame
    
    def run(self):
        """
        Run the streaming application
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("\n🎥 Webcam streaming started!")
        print("Controls:")
        print("  'p' - Pause/Resume OSC streaming")
        print("  'q' - Quit application")
        print("\nTracking body and streaming to OSC/VMC...")
        print("Virtual joints (VT1, SACROILIAC) are included in the stream!")
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Failed to capture frame")
                    continue
                
                # Process and stream
                annotated_frame = self.process_and_stream(frame)
                
                # Display
                cv2.imshow('MediaPipe → OSC/VMC Stream (with Virtual Joints)', 
                          annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('p'):
                    self.streaming_enabled = not self.streaming_enabled
                    status = "resumed" if self.streaming_enabled else "paused"
                    print(f"📡 OSC streaming {status}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n✅ Total frames streamed: {self.osc_client.frame_count}")
            print("👋 Goodbye!")


def main():
    """
    Main entry point
    """
    print("\n" + "="*70)
    print("MediaPipe ENHANCED → OSC/VMC Streamer")
    print("With Virtual Joints Support (VT1 & SACROILIAC)")
    print("="*70)
    
    # Configuration
    print("\nConfiguration:")
    print("1. Use default settings (localhost:39539, virtual joints ON)")
    print("2. Custom settings")
    
    choice = input("\nChoose option (1 or 2, default=1): ").strip()
    
    if choice == "2":
        osc_ip = input("Enter OSC server IP (default: 127.0.0.1): ").strip() or "127.0.0.1"
        osc_port = input("Enter OSC server port (default: 39539): ").strip()
        osc_port = int(osc_port) if osc_port else 39539
        
        include_virtual = input("Include virtual joints VT1/SACROILIAC? (y/n, default=y): ").strip().lower()
        include_virtual_joints = include_virtual != 'n'
    else:
        osc_ip = "127.0.0.1"
        osc_port = 39539
        include_virtual_joints = True
    
    print("\n" + "="*70)
    print("Starting streamer with settings:")
    print(f"  OSC Server: {osc_ip}:{osc_port}")
    print(f"  Virtual Joints: {'✅ Enabled' if include_virtual_joints else '❌ Disabled'}")
    print("="*70)
    
    # Create and run streamer
    streamer = MediaPipeOSCStreamer(
        osc_ip=osc_ip,
        osc_port=osc_port,
        include_virtual_joints=include_virtual_joints
    )
    
    streamer.run()


if __name__ == "__main__":
    main()
