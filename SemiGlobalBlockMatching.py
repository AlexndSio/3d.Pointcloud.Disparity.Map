import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Images
left_image = cv2.imread(r'C:\Users\Alex\Desktop\3d\000006_10.png', 0)
right_image = cv2.imread(r'C:\Users\Alex\Desktop\3d\000006_11.png', 0)

# Denoising
left_image = cv2.GaussianBlur(left_image, (5,5), 0)
right_image = cv2.GaussianBlur(right_image, (5,5), 0)

# Disparity map
def dispar_map(imgL, imgR):

    window_size = 3

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=16*10,
        blockSize=window_size,
        P1=9 * 3 * window_size,
        P2=128 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=40,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # Filter
    lmbda = 100000
    sigma = 2
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR)
    dispr = right_matcher.compute(imgR, imgL)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    return filteredImg

def visualize_point_cloud(pc, colors=None):
    pc = o3d.utility.Vector3dVector(pc)
    pc = o3d.geometry.PointCloud(pc)
    if colors is not None:
        colors = np.array(colors).reshape(-1, 1)
        colors = np.repeat(colors, 3, axis=1)
        pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pc])

# Denoising
disparity_map = dispar_map(left_image, right_image)
disparity_map = cv2.medianBlur(disparity_map, 7)

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
        color = left_image[j,i]
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

plt.imshow(disparity_map, cmap='gray')
plt.title('Disparity Map')
plt.show()
visualize_point_cloud(vertices, colors)