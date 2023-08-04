import cv2
import numpy as np
import open3d as o3d

def visualize_point_cloud(pc, colors=None):
    pc = o3d.utility.Vector3dVector(pc)
    pc = o3d.geometry.PointCloud(pc)
    if colors is not None:
        colors = np.array(colors).reshape(-1, 1)
        colors = np.repeat(colors, 3, axis=1)  
        pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pc])

# Images
left_frame = cv2.imread("000006_10.png", cv2.IMREAD_GRAYSCALE)
right_frame = cv2.imread("000006_11.png", cv2.IMREAD_GRAYSCALE)

# Denoising
left_frame = cv2.fastNlMeansDenoising(left_frame, None, 10, 7, 21)
right_frame = cv2.fastNlMeansDenoising(right_frame, None, 10, 7, 21)

# Initializations
disparity_map = np.zeros_like(left_frame)
block_size = 8
max_num_blocks = 50
frame_shape = left_frame.shape

# Parsing
for i in range(frame_shape[0] - block_size):
    for j in range(frame_shape[1] - block_size):
        template = left_frame[i:i+block_size, j:j+block_size]

        start = max(j - max_num_blocks * block_size, 0)
        end = j + block_size
        
        roi = right_frame[i:i+block_size, start:end]

        res = cv2.matchTemplate(roi, template, cv2.TM_SQDIFF)

        _, _, min_loc, _ = cv2.minMaxLoc(res)

        disparity = j - (min_loc[0] + start)

        disparity_map[i, j] = disparity



# Initializations for the 3d model
rows, columns = disparity_map.shape
u0 = columns / 2
v0 = rows / 2
stereo_baseline = 0.2
fieldOfView = 1.2
focal_length_constant = 1.0 / (2.0 * np.tan(fieldOfView / 2.0))

vertices = []
colors = []
# Poinctloud
for i in range(columns):
    for j in range(rows):
        d = disparity_map[j, i] / columns
        color = left_frame[j, i]
        z = fieldOfView * stereo_baseline / d

        if d == 0:
            continue
        x = ((i - u0) / columns) * (z / focal_length_constant)
        y = -((j - v0) / rows) * (z / focal_length_constant)

        if z > 0:
            vertices.append([40 * x, 40 * y, -40 * z])
            colors.append(np.array(color) / 255.0)

vertices = np.stack(vertices)
colors = np.stack(colors)

visualize_point_cloud(vertices, colors)