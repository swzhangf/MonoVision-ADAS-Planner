import cv2
import numpy as np
import requests
import os
import sys
from ultralytics import YOLO
# Assuming src.transform, src.cubic_spline, and src.frenet_optimal exist
from src.transform import PerspectiveTransformer
from src.cubic_spline import CubicSpline2D
from src.frenet_optimal import frenet_optimal_planning

def download_assets():
    assets_dir = "assets"
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    
    save_path = os.path.join(assets_dir, "road_video.mp4")
    # This is an official test video from the Udacity autonomous driving course.
    # The video content includes a highway, lane lines, and other vehicles, making it realistic.
    url = "https://github.com/udacity/CarND-Advanced-Lane-Lines/raw/master/project_video.mp4"

    if os.path.exists(save_path):
        if os.path.getsize(save_path) < 100 * 1024:
            print("Detected old/corrupted video file, deleting and retrying...")
            os.remove(save_path)
        else:
            print(f"Video file is ready: {save_path}")
            return save_path

    print(f"Downloading test video from the website...")
    try:
        r = requests.get(url, stream=True, timeout=30)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk: f.write(chunk)
            print("Video downloaded successfully!")
            return save_path
        else:
            print(f"Download failed, status code: {r.status_code}")
    except Exception as e:
        print(f"Download error (network issue): {e}")
    
    print("Automatic download failed.")
    print("Please manually download this link: " + url)
    print(f"And rename the file to road_video.mp4 and place it in the {os.path.abspath(assets_dir)} folder.")
    sys.exit(1)

def main():
    video_path = download_assets()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot read video file. Please check the path.")
        return
    print("Loading YOLOv8n...")
    model = YOLO('yolov8n.pt') 
    transformer = PerspectiveTransformer()
    
    # Generate reference path
    # The path here needs to be slightly longer than the visible distance in the video
    wx = np.linspace(0, 100, 10)
    wy = np.zeros_like(wx)
    csp = CubicSpline2D(wx, wy)
    
    # 3. Vehicle Initial State
    c_speed = 12.0  # m/s (approx 43km/h)
    c_d = 0.0       # Lateral offset
    c_d_d = 0.0
    c_d_dd = 0.0
    s0 = 0.0        # Longitudinal distance
    
    print("\nAutonomous Driving Simulation System Started")
    print("Press 'q' to exit the program")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video when playback is finished
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            s0 = 0 # Reset planning distance
            continue
            
        frame_count += 1
        if frame_count % 3 != 0:
            cv2.imshow('ADAS Planning System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # Resize to speed up processing
        frame = cv2.resize(frame, (1280, 720))
        
        # Perception
        results = model(frame, classes=[2, 5, 7], device='cpu', verbose=False, conf=0.3)
        
        obstacles_pixel = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy() 
                cx = (x1 + x2) / 2
                cy = y2 
                obstacles_pixel.append([cx, cy])
                # Draw green bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Mapping
        obstacles_bev = transformer.transform_points_to_bev(obstacles_pixel)
        
        valid_obs = []
        if len(obstacles_bev) > 0:
            for ob in obstacles_bev:
                # Filtering logic: Only care about vehicles 0-50m ahead and within 8m laterally
                if 0 < ob[0] < 50 and abs(ob[1]) < 8:
                    valid_obs.append(ob)
        valid_obs = np.array(valid_obs)

        # Planning
        path = frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, valid_obs)
        
        if path is not None:
            # Update virtual vehicle state
            s0 = path.s[1] 
            c_d = path.d[1]
            c_d_d = path.d_d[1]
            c_d_dd = path.d_dd[1]
            c_speed = path.s_d[1]
            
            if s0 > 80: s0 = 0 # Reset s0 if it goes too far
            
            # Visualization
            display_x, display_y = [], []
            for i in range(len(path.x)):
                # Convert planned coordinates back to relative coordinates
                dx = path.x[i] - path.x[0] 
                dy = path.y[i]
                if dx >= 0: 
                    display_x.append(dx)
                    display_y.append(dy)
            
            # Draw line only if there are enough points
            if len(display_x) > 2:
                traj_pixels = transformer.transform_path_to_image(display_x, display_y)
                if len(traj_pixels) > 1:
                    # Draw the planned trajectory in red, thicker line
                    cv2.polylines(frame, [traj_pixels], False, (0, 0, 255), 4)
        
        # Display status on screen
        text_speed = f"Ego Speed: {c_speed*3.6:.1f} km/h"
        text_obs = f"Obstacles Detected: {len(valid_obs)}"
        cv2.putText(frame, text_speed, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, text_obs, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('ADAS Planning System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()