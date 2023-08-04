import cv2 
import numpy as np
import open3d as o3d

# Images
left_frame = cv2.imread("000006_10.png", cv2.IMREAD_GRAYSCALE)
right_frame = cv2.imread("000006_11.png", cv2.IMREAD_GRAYSCALE)
disparity_map = cv2.imread('000006_10disp.png')[..., 0]

def visualize_point_cloud(pc, colors=None):
    pc = o3d.utility.Vector3dVector(pc)
    pc = o3d.geometry.PointCloud(pc)
    if colors is not None:
        colors = np.array(colors).reshape(-1, 1)
        colors = np.repeat(colors, 3, axis=1)
        pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pc])

# Initializations
rows, columns = disparity_map.shape[0], disparity_map.shape[1]
u0 = columns / 2 
v0 = rows / 2
stereo_baseline = 0.2
fieldOfView = 1.2
focal_length_constant = 1.0 / (2.0 * np.tan(fieldOfView / 2.0))

vertices = []
colors = []

# Pointcloud
for i in range(columns):
    for j in range(rows):

        d = disparity_map[j,i]/columns
        color = left_frame[j,i]
        z = fieldOfView * stereo_baseline /d 

        if d == 0 : 
            continue
        x = ((i- u0)/ columns ) * (z/focal_length_constant)
        y = -((j- v0)/ rows ) * (z/focal_length_constant)

        if z >0: 
            vertices.append([40*x,40*y, -40 *z])
            colors.append(np.array(color)/255.0)

vertices = np.stack(vertices)
colors = np.stack(colors)

visualize_point_cloud(vertices, colors)