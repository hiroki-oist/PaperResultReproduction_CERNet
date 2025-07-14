import numpy as np
import time
import argparse
import os

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

def move_to_home(reachy):
    home_position = np.array([
      [-0.029,  0.284, -0.959,  0.346],
      [ 0.024,  0.959,  0.283,  0.247],  
      [ 0.999, -0.015, -0.035, -0.319],
      [0, 0, 0, 1],  
    ])
    
    joint_pos_home = reachy.l_arm.inverse_kinematics(home_position)
    
    print("Moving to home position...")
    reachy.turn_on('l_arm')
    goto({joint: pos for joint, pos in zip(reachy.l_arm.joints.values(), joint_pos_home)}, duration=1.0, interpolation_mode=InterpolationMode.MINIMUM_JERK)
    
    # 🆕 グリッパーの角度を10度に設定
    if hasattr(reachy.l_arm, 'l_gripper'):
        print("Setting gripper to 10 degrees...")
        reachy.l_arm.l_gripper.goal_position = 10.0
    else:
        print("Warning: l_gripper not found on reachy.l_arm!")

def collect_data(save_directory, num_seq, seq_len, fps):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Connect to Reachy
    reachy = ReachySDK(host='10.42.0.1')
    
    # Initialize the data array
    data = np.empty((num_seq, seq_len, 3))
    
    interval = 1.0 / fps  # Time interval between data collection
    
    for seq in range(num_seq):
        move_to_home(reachy)
        os.system("afplay /System/Library/Sounds/Glass.aiff")
        input(f"Press Enter to start measuring sequence {seq + 1}/{num_seq}...")
        print(f"Collecting sequence {seq+1}/{num_seq}")
        time.sleep(3)
        
        for t in range(seq_len):
            print("Recording step: ", t)
            for joint_name, joint in reachy.l_arm.joints.items():
                if joint_name != 'l_gripper':
                    joint.compliant = True
            
            start_time = time.time()  # Record the start time of the loop
            
            fk_matrix = reachy.l_arm.forward_kinematics()
            end_effector_pos = fk_matrix[:3, 3]  # Extract x, y, z coordinates
            
            data[seq, t] = end_effector_pos
            
            elapsed_time = time.time() - start_time  # Measure processing time
            sleep_time = max(0, interval - elapsed_time)  # Calculate required sleep time
            time.sleep(sleep_time)
        
        os.system("afplay /System/Library/Sounds/Glass.aiff")
        reachy.turn_on('l_arm')
        
        if hasattr(reachy.l_arm, 'l_gripper'):
            reachy.l_arm.l_gripper.goal_position = 7.0
    
    # Save data to file
    filename = os.path.join(save_directory, f"{num_seq}_{seq_len}_3.npy")
    np.save(filename, data)
    print(f"Data saved to {filename}")
    reachy.turn_off_smoothly('l_arm')

def main():
    parser = argparse.ArgumentParser(description="Collect Reachy End-Effector Position Data")
    parser.add_argument("--save_directory", type=str, required=True, help="Directory to save the output .npy file")
    parser.add_argument("--num_seq", type=int, default=1, help="Number of sequences to collect")
    parser.add_argument("--seq_len", type=int, default=100, help="Length of each sequence")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second (sampling rate)")
    args = parser.parse_args()
    
    collect_data(args.save_directory, args.num_seq, args.seq_len, args.fps)

if __name__ == "__main__":
    main()
