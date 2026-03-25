import os
import cv2
import numpy as np
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# --- LINUX GUI FIX ---
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

# ArUco Marker ID Dictionary (KMUTNB RE Lab)
ARUCO_MEANINGS = {
    0:  "Home Position / Origin",
    1:  "Pick-up Zone A",
    2:  "Pick-up Zone B",
    3:  "Drop-off Zone",
    4:  "Obstacle Warning",
    23: "Waypoint 23",
}

def get_handle(sim, possible_names, fallback_id=None):
    """Tries multiple names, and finally a numeric ID fallback."""
    for name in possible_names:
        try:
            handle = sim.getObject(name)
            if handle != -1: return handle
        except:
            continue
    # If name search fails, use the known numeric handle from Part_Name.txt
    return fallback_id

def get_image(sim, sensor_handle):
    try:
        img, resolution = sim.getVisionSensorImg(sensor_handle)
        if not img or resolution[0] == 0: return None
        arr = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
        return cv2.flip(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), 0)
    except: 
        return None

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
            c = corners[i][0]
            cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
            cv2.putText(frame, meaning, (cx - 40, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, found_now

def main():
    print("Connecting to CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    print("Starting Simulation...")
    sim.startSimulation() 
    time.sleep(1) 

    # --- OBJECT DISCOVERY ---
    # Using fallback IDs from your Part_Name.txt list
    sensor_handle = get_handle(sim, ['./Vision_sensor', 'Vision_sensor'], fallback_id=384)
    tcp_handle = get_handle(sim, ['./gripperEF', 'gripperEF'], fallback_id=93)
    
    # Targeting the cup base pose directly (Handle 170 in your text file)
    cup_handle = get_handle(sim, ['./cup_pose#0', 'cup_pose#0', 'cup_pose'], fallback_id=170)

    if None in [sensor_handle, tcp_handle, cup_handle]:
        print("✘ Error: Still could not find one or more objects.")
        print(f"Status: Sensor={sensor_handle}, TCP={tcp_handle}, Cup={cup_handle}")
        return
    
    print(f"✔ Success! Connected to Sensor({sensor_handle}), TCP({tcp_handle}), and Cup({cup_handle})")

    # ArUco Setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    WINDOW_NAME = "RE-LAB ArUco Vision"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    print("\n" + "="*80)
    print(f"{'KMUTNB RE-LAB: TCP & CUP BASE TRACKING':^80}")
    print("="*80)

    try:
        while True:
            # TCP = End-effector Position | CC = Center of Cup (Base)
            TCP = sim.getObjectPosition(tcp_handle, -1)
            CC = sim.getObjectPosition(cup_handle, -1)

            # Distance calculation for accuracy
            dist = np.linalg.norm(np.array(TCP) - np.array(CC))

            # Display coordinates
            print(f"\rTCP: [{TCP[0]:.3f}, {TCP[1]:.3f}, {TCP[2]:.3f}] | "
                  f"CC: [{CC[0]:.3f}, {CC[1]:.3f}, {CC[2]:.3f}] | "
                  f"Dist: {dist:.3f}m", end="")

            # Vision Window
            frame = get_image(sim, sensor_handle)
            if frame is not None:
                annotated, _ = detect_and_annotate(frame, detector)
                cv2.imshow(WINDOW_NAME, annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        print("\n\nProcess terminated.")

if __name__ == "__main__":
    main()