#!/usr/bin/env python3
# live_pose_capture.py
import time
import signal
import sys
from pathlib import Path
import numpy as np
from reachy_sdk import ReachySDK

HOST = '10.42.0.1'     
INTERVAL = 5.0        
SAVE_PATH = Path('home_position.npy')  

def main():
    reachy = ReachySDK(host=HOST)

    reachy.turn_off_smoothly('l_arm')
    time.sleep(3)

    def _cleanup(signum=None, frame=None):
        print('\n[INFO] Turning torque ON & quitting…')
        reachy.turn_on('l_arm')
        sys.exit(0)
    signal.signal(signal.SIGINT, _cleanup)

    while True:
        start = time.time()
        pose = reachy.l_arm.forward_kinematics()  # 4×4 numpy.array
        print(np.array2string(pose, precision=3, suppress_small=False))

        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            _ = sys.stdin.readline()
            print('\n[SAVED] Saved to →', SAVE_PATH.resolve())
            np.save(SAVE_PATH, pose)
            _cleanup()

        elapsed = time.time() - start
        time.sleep(max(0, INTERVAL - elapsed))

if __name__ == '__main__':
    import select
    if sys.platform.startswith('win'):
        import msvcrt
        select.select = lambda r, w, x, t=0: ([msvcrt.kbhit() and sys.stdin] if msvcrt.kbhit() else [], [], [])

    main()
