import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(self):
        # We need to hard-code a trapezoidal Region of Interest (ROI) here
        # This region corresponds to a rectangular area of the road surface in the video
        # Adapted for the video we are going to download
        
        # Assume video resolution is 1280x720 (The video we download is roughly this ratio)
        self.src_points = np.float32([
            [550, 450],   # Top-left (far end of the road)
            [730, 450],   # Top-right (far end of the road)
            [1280, 680],  # Bottom-right (near end of the road)
            [0, 680]      # Bottom-left (near end of the road)
        ])

        # Dimensions of the transformed Bird's Eye View (BEV) image (God's view)
        # Assume we map to 0-40 meters distance, -10 to 10 meters width
        self.dst_points = np.float32([
            [0, 0],       # Top-left (Far end x=40m, y=-10m) -> Mapping logic needs adjustment
            [200, 0],     # Top-right
            [200, 600],   # Bottom-right
            [0, 600]      # Bottom-left
        ])
        
        # Calculate the transformation matrix
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        
        # Scale factor from pixels to meters (rough estimate)
        # Assume BEV image height of 600 pixels represents 30 meters
        self.pixels_per_meter_y = 600 / 30.0 
        self.pixels_per_meter_x = 200 / 10.0 # Assume road width is 10 meters

    def transform_image(self, img):
        # Transform the image into a Bird's Eye View
        h, w = img.shape[:2]
        warped = cv2.warpPerspective(img, self.M, (200, 600))
        return warped

    def transform_points_to_bev(self, points):
        """
        Convert image pixel coordinates (u, v) to Vehicle Coordinate System (x, y)
        :param points: List of [x, y] in image
        :return: List of [x, y] in meters (Vehicle Frame: x forward, y left)
        """
        if len(points) == 0: return np.array([])
        
        # Add a dimension for matrix multiplication
        pts = np.float32(points).reshape(-1, 1, 2)
        dst_pts = cv2.perspectiveTransform(pts, self.M)
        dst_pts = dst_pts.reshape(-1, 2)
        
        # Convert to meter coordinate system (origin at the current vehicle position)
        # BEV image coordinates: (0,0) is top-left, x right, y down
        # Vehicle coordinates: x forward, y left
        
        real_world_points = []
        img_h = 600
        img_w = 200
        
        for px, py in dst_pts:
            # Image y-axis is down, Vehicle x-axis is forward -> x_meter = (img_h - py) / ppm_y
            x_meter = (img_h - py) / self.pixels_per_meter_y
            # Image x-axis is right, Vehicle y-axis is left -> y_meter = (img_w/2 - px) / ppm_x
            y_meter = (img_w / 2 - px) / self.pixels_per_meter_x
            real_world_points.append([x_meter, y_meter])
            
        return np.array(real_world_points)

    def transform_path_to_image(self, path_x, path_y):
        
        # Project the planned trajectory (in meters) back to video pixels
        
        if len(path_x) == 0: return []
        
        img_pts = []
        img_h = 600
        img_w = 200
        
        # Meters -> BEV Pixels
        for x, y in zip(path_x, path_y):
            # Vehicle x forward -> Image y down
            py = img_h - (x * self.pixels_per_meter_y)
            # Vehicle y left -> Image x right
            px = (img_w / 2) - (y * self.pixels_per_meter_x)
            img_pts.append([px, py])
            
        # BEV Pixels -> Original Video Pixels
        pts = np.float32(img_pts).reshape(-1, 1, 2)
        orig_pts = cv2.perspectiveTransform(pts, self.M_inv)
        return orig_pts.reshape(-1, 2).astype(int)