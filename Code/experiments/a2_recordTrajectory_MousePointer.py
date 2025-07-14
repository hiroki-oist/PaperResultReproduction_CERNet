import numpy as np
import time
import pyautogui
import os
import sys
import threading
import queue
from scipy.interpolate import interp1d

# === パラメータ設定 ===
NUM_SEQUENCES    = 1            # number of sequences to record
TARGET_STEPS     = 100           # number of timesteps to save
OUTPUT_PATH      = "a.npy"  # path to save the data
PRE_START_WAIT   = 3            # number of seconds to wait before recording

# ======================
stop_recording = False
queue_data = queue.Queue()

def play_sound(sound_type):
    if sys.platform.startswith('darwin'):
        if sound_type == 'start':
            os.system("afplay /System/Library/Sounds/Ping.aiff")
        elif sound_type == 'end':
            os.system("afplay /System/Library/Sounds/Glass.aiff")
    elif sys.platform.startswith('win'):
        import winsound
        if sound_type == 'start':
            winsound.Beep(1000, 300)
        elif sound_type == 'end':
            winsound.Beep(500, 300)
    else:
        print('\a', end='')

def record_sequence():
    global stop_recording
    seq_data = []
    timestamps = []
    start_time = time.time()
    print("Starting recording... press enter key to end")
    
    while not stop_recording:
        pos = pyautogui.position()
        current_time = time.time() - start_time
        seq_data.append([pos.x, pos.y])
        timestamps.append(current_time)
        queue_data.put(len(seq_data)) 
        time.sleep(0.01) 
    return np.array(seq_data), np.array(timestamps)

def interpolate_sequence(seq_data, timestamps):
    if len(seq_data) < 2:
        return np.tile(seq_data, (TARGET_STEPS, 1))
    
    new_timestamps = np.linspace(timestamps[0], timestamps[-1], TARGET_STEPS)
    interp_func_x = interp1d(timestamps, seq_data[:, 0], kind='linear', fill_value='extrapolate')
    interp_func_y = interp1d(timestamps, seq_data[:, 1], kind='linear', fill_value='extrapolate')
    
    new_seq_data = np.vstack((interp_func_x(new_timestamps), interp_func_y(new_timestamps))).T
    return new_seq_data

def main():
    global stop_recording
    print(f"n_seq: {NUM_SEQUENCES}, n_timesteps: {TARGET_STEPS}")
    print(f"Data will be stored at {OUTPUT_PATH} .")

    data = []
    
    for seq in range(NUM_SEQUENCES):
        input(f"\nSequence number {seq+1} starting. Press enter...")
        print(f"Waiting for {PRE_START_WAIT} seconds...")
        time.sleep(PRE_START_WAIT)
        stop_recording = False
        record_thread = threading.Thread(target=lambda: data.append(record_sequence()))
        record_thread.start()
        print("Recording started ! Press enter key to finish recording")
        play_sound("start")
        input("Press enter...")
        stop_recording = True
        record_thread.join()
        
        seq_data, timestamps = data[-1]
        interpolated_data = interpolate_sequence(seq_data, timestamps)
        data[-1] = interpolated_data

        play_sound("end")
        print(f"Sequence {seq+1} Finish.")
    
    try:
        np.save(OUTPUT_PATH, np.array(data))
        print(f"\nAll {NUM_SEQUENCES} sequences saved to {OUTPUT_PATH}.")
    except Exception as e:
        print("Error saving file:", e)

if __name__ == "__main__":
    main()
