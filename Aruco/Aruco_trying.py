import os
import cv2
import numpy as np
import time

# --- LINUX GUI FIX ---
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ArUco Marker ID Dictionary (KMUTNB RE Lab)
ARUCO_MEANINGS = {
    0:  "Home Position / Origin",
    1:  "Pick-up Zone A",
    2:  "Pick-up Zone B",
    3:  "Drop-off Zone",
    4:  "Obstacle Warning",
    23: "Waypoint 23",
}

def get_image(sim, sensor_handle):
    try:
        img, resolution = sim.getVisionSensorImg(sensor_handle)
        if not img or resolution[0] == 0: return None
        # Convert and flip (CoppeliaSim origin is bottom-left)
        arr = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
        return cv2.flip(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), 0)
    except: return None

def detect_and_annotate(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    found_now = []

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            mid = int(marker_id)
            meaning = ARUCO_MEANINGS.get(mid, f"Unknown (ID:{mid})")
            found_now.append((mid, meaning))

            # Draw label on the video frame
            c = corners[i][0]
            cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
            cv2.putText(frame, meaning, (cx - 40, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, found_now

def main():
    print("Connecting to CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    # --- AUTOMATIC SIMULATION START ---
    print("Starting Simulation...")
    sim.startSimulation() 
    time.sleep(1) # Give the sensor a second to initialize

    # Get Vision Sensor
    try:
        sensor_handle = sim.getObject('/Vision_sensor')
        print(f"✔ Connected to Vision_sensor")
    except:
        print("✘ Error: Could not find '/Vision_sensor'. Check your .ttt file.")
        return

    # ArUco Setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    # --- DISPLAY SETTINGS ---
    WINDOW_NAME = "RE-LAB ArUco Vision"
    SCALE_FACTOR = 2.0  # Set to 2.0 for 2x larger screen
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    print("\n" + "="*45)
    print(f"{'MARKER LOGGING ACTIVE':^45}")
    print("="*45)

    last_seen_ids = set()

    try:
        while True:
            frame = get_image(sim, sensor_handle)
            if frame is not None:
                # 1. Detection
                annotated, detected_list = detect_and_annotate(frame, detector)

                # 2. Terminal Print (Only for NEW detections)
                current_ids = {m[0] for m in detected_list}
                new_ids = current_ids - last_seen_ids
                for mid, meaning in detected_list:
                    if mid in new_ids:
                        print(f"[*] SEEN: ID {mid: <2} | {meaning}")
                last_seen_ids = current_ids

                # 3. Resize window
                w, h = int(annotated.shape[1] * SCALE_FACTOR), int(annotated.shape[0] * SCALE_FACTOR)
                bigger_frame = cv2.resize(annotated, (w, h), interpolation=cv2.INTER_LINEAR)

                cv2.imshow(WINDOW_NAME, bigger_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Optional: Stop simulation when you close the script
        # sim.stopSimulation() 
        cv2.destroyAllWindows()
        print("\nSimulation stopped. Program closed.")

if __name__ == "__main__":
    main()