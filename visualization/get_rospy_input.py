import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
import cv2
import numpy as np

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

def rgb_callback(msg, index):
    global rgb_images
    image = ros_compressed_image_to_cv_image(msg, encoding=cv2.IMREAD_COLOR)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_images[index] = image

def depth_callback(msg, index):
    global depth_images
    image = ros_compressed_image_to_cv_image(msg, encoding=cv2.IMREAD_UNCHANGED)
    image = image.astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = image / 1000.0
    depth_images[index] = image

def camera_info_callback(msg, index):
    global camera_infos
    K = np.array(msg.K).reshape(3, 3)
    camera_infos[index] = K

for i in range(3):
    rospy.Subscriber(f'/cam_{i+1}/rgb/compressed', CompressedImage, rgb_callback, callback_args=i)
    rospy.Subscriber(f'/cam_{i+1}/depth/compressed', CompressedImage, depth_callback, callback_args=0)
    rospy.Subscriber(f'/cam_{i+1}/rgb/camera_info', CameraInfo, camera_info_callback, callback_args=0)

while not rospy.is_shutdown():
    rospy.sleep(0.1)