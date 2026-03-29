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
}

def get_handle(sim, possible_names, fallback_id=None):
    """Tries multiple names, and finally a numeric ID fallback."""
    for name in possible_names:
        try:
            handle = sim.getObject(name)
            if handle != -1: return handle
        except:
            continue
    return fallback_id

def get_image(sim, sensor_handle):
    """Captures and converts the vision sensor image from the simulation."""
    try:
        img, resolution = sim.getVisionSensorImg(sensor_handle)
        if not img or resolution[0] == 0: return None
        arr = np.frombuffer(img, dtype=np.uint8).reshape((resolution[1], resolution[0], 3))
        # Flip and convert RGB to BGR for OpenCV
        return cv2.flip(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), 0)
    except: 
        return None

def detect_and_annotate(frame, detector):
    """Detects ArUco markers, draws labels, and calculates pixel centers."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    found_markers = {} # Dictionary to store center points
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            mid = int(marker_id)
            
            # Calculate the center of the square in pixels
            c = corners[i][0]
            cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
            
            # Store the coordinates
            found_markers[mid] = (cx, cy)
            
            # Annotate the image
            meaning = ARUCO_MEANINGS.get(mid, f"ID:{mid}")
            cv2.putText(frame, f"{meaning} ({cx},{cy})", (cx - 40, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
    return frame, found_markers

def main():
    print("Connecting to CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    print("Starting Simulation...")
    sim.startSimulation() 
    time.sleep(1) 

    # --- OBJECT DISCOVERY ---
    sensor_handle = get_handle(sim, ['./Vision_sensor', 'Vision_sensor'], fallback_id=384)

    # Verify we found the sensor
    if sensor_handle is None or sensor_handle == -1:
        print("✘ Error: Could not find Vision Sensor.")
        return

    # ArUco Setup for Vision Window
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    WINDOW_NAME = "RE-LAB ArUco Tracking"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    print("\n" + "="*80)
    print(f"{'KMUTNB RE-LAB: ARUCO SCREEN COORDINATES (PIXELS)':^80}")
    print("="*80)

    try:
        while True:
            # 1. Capture Image
            frame = get_image(sim, sensor_handle)
            if frame is not None:
                # 2. Get annotated frame and dictionary of pixel centers
                annotated, marker_centers = detect_and_annotate(frame, detector)
                cv2.imshow(WINDOW_NAME, annotated)

                # 3. Build the print string based on what the CAMERA sees
                status_line = "\r"
                valid_pts = []
                
                # Check specifically for IDs 1, 2, 3, and 4
                for m_id in [1, 2, 3, 4]:
                    if m_id in marker_centers:
                        pos = marker_centers[m_id]
                        status_line += f"ID{m_id}:[{pos[0]}, {pos[1]}]  "
                        valid_pts.append(pos)
                    else:
                        status_line += f"ID{m_id}:[NOT IN VIEW]  "

                # 4. Calculate geometric center of the 4 markers on the screen
                if len(valid_pts) == 4:
                    avg_x = sum(p[0] for p in valid_pts) // 4
                    avg_y = sum(p[1] for p in valid_pts) // 4
                    status_line += f"| Screen Center: [{avg_x}, {avg_y}]"
                    
                    # Optional: Draw a red circle at the calculated center on the image
                    cv2.circle(annotated, (avg_x, avg_y), 5, (0, 0, 255), -1)
                    cv2.putText(annotated, "CENTER", (avg_x + 10, avg_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imshow(WINDOW_NAME, annotated) # Update window with center dot

                # Print coordinates to console in a single updating line
                print(status_line, end="", flush=True)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        sim.stopSimulation()
        cv2.destroyAllWindows()
        print("\n\nSimulation stopped and process terminated.")

if __name__ == "__main__":
    main()