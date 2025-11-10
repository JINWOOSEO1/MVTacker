import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
import cv2
import numpy as np
import os

def ros_compressed_image_to_cv_image(
    msg: CompressedImage, encoding=cv2.IMREAD_UNCHANGED
):
    # Decompress the image data, jpeg byte array to cv2 image
    np_arr = np.fromstring(msg.data, np.uint8)
    if np_arr.size == 0:
        rospy.logerr("Received empty image data")
        return None
    cv_image = cv2.imdecode(np_arr, encoding)
    return cv_image

rospy.init_node('input_node', anonymous=True)

rgb_images = [None, None, None]
depth_images = [None, None, None]
camera_infos = [None, None, None]
rgb_idx, depth_idx, intr_idx = 0, 0, 0

def rgb_callback(msg, index):
    global rgb_images
    global rgb_idx
    image = ros_compressed_image_to_cv_image(msg, encoding=cv2.IMREAD_COLOR)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_images[index] = image
    if(index is 0 and rgb_idx==0):
        cv2.imwrite("../../datasets/extrinsics/rgb_1.png", image)
        rgb_idx += 1


def depth_callback(msg, index):
    global depth_images
    global depth_idx
    image = ros_compressed_image_to_cv_image(msg, encoding=cv2.IMREAD_UNCHANGED)
    image = image.astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = image / 1000.0
    depth_images[index] = image
    if(index is 0 and depth_idx==0):
        cv2.imwrite("../../datasets/extrinsics/depth_1.png", image) 
        depth_idx += 1

def camera_info_callback(msg, index):
    global camera_infos
    global intr_idx
    K = np.array(msg.K).reshape(3, 3)
    camera_infos[index] = K
    if(index is 0 and intr_idx==0):
        np.save("../../datasets/extrinsics/intrinsics_1.npy", K)
        intr_idx += 1

for i in range(3):
    rospy.Subscriber(f'/cam_{i+1}/rgb/compressed', CompressedImage, rgb_callback, callback_args=i)
    rospy.Subscriber(f'/cam_{i+1}/depth/compressed', CompressedImage, depth_callback, callback_args=i)
    rospy.Subscriber(f'/cam_{i+1}/rgb/camera_info', CameraInfo, camera_info_callback, callback_args=i)

while not rospy.is_shutdown():
    rospy.sleep(0.1)