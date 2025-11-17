from argparse import ArgumentParser
import cv2
import numpy as np

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo

main_camera_index = 0
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

class DataCaptureNode:
    def __init__(self, num_cameras=1, crop_regions_path=None):
        self.window_len = 12
        self.num_new_img = 0

        self.num_cameras = num_cameras

        self.rgb_subscribers = []
        self.depth_subscribers = []
        self.camera_info_subscribers = []

        self.rgb_images = [[] for _ in range(num_cameras)]
        self.depth_images = [[] for _ in range(num_cameras)]
        self.camera_infos = [[] for _ in range(num_cameras)]

        self.crop_regions = None
        if crop_regions_path is not None:
            self.crop_regions = np.load(crop_regions_path, allow_pickle=True)

        for i in range(num_cameras):
            rgb_topic = f"/cam_{i+1}/rgb/compressed"
            depth_topic = f"/cam_{i+1}/depth/compressed"
            camera_info_topic = f"/cam_{i+1}/rgb/camera_info"

            self.rgb_subscribers.append(
                rospy.Subscriber(
                    rgb_topic, CompressedImage, self.rgb_callback, callback_args=i
                )
            )
            self.depth_subscribers.append(
                rospy.Subscriber(
                    depth_topic, CompressedImage, self.depth_callback, callback_args=i
                )
            )
            self.camera_info_subscribers.append(
                rospy.Subscriber(
                    camera_info_topic,
                    CameraInfo,
                    self.camera_info_callback,
                    callback_args=i,
                )
            )

    def rgb_callback(self, msg, camera_index):
        cv_image = ros_compressed_image_to_cv_image(msg, cv2.IMREAD_COLOR)

        if self.crop_regions is not None:
            x1, y1, x2, y2 = self.crop_regions[camera_index]
            cv_image = cv_image[y1:y2, x1:x2]

        TARGET_SIZE = (640, 360) 
        if cv_image.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
            cv_image = cv2.resize(cv_image, TARGET_SIZE, 
                                interpolation=cv2.INTER_LINEAR)
            
        self.rgb_images[camera_index].append(cv_image.astype(np.uint8))
        if(len(self.rgb_images[camera_index]) > self.window_len):
            self.rgb_images[camera_index].pop(0)
        if camera_index == main_camera_index:
            self.num_new_img = self.num_new_img + 1

    def depth_callback(self, msg, camera_index):
        cv_image = ros_compressed_image_to_cv_image(msg, cv2.IMREAD_UNCHANGED)

        if self.crop_regions is not None:
            x1, y1, x2, y2 = self.crop_regions[camera_index]
            cv_image = cv_image[y1:y2, x1:x2]

        TARGET_SIZE = (640, 360) 
        if cv_image.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
            cv_image = cv2.resize(cv_image, TARGET_SIZE, 
                                interpolation=cv2.INTER_NEAREST)

        self.depth_images[camera_index].append(cv_image)
        if(len(self.depth_images[camera_index]) > self.window_len):
            self.depth_images[camera_index].pop(0)

    def camera_info_callback(self, msg: CameraInfo, camera_index):
        if len(self.camera_infos[camera_index]) is not self.window_len:
            K = list(msg.K)
            cx = K[2]
            cy = K[5]
            if self.crop_regions is not None:
                x1, y1, x2, y2 = self.crop_regions[camera_index]
                cx -= x1
                cy -= y1
                K[2] = cx
                K[5] = cy
            # Scaling for resizing to (640, 360 )
            K[0] = K[0]/2
            K[4] = K[4]/2
            K[2] = K[2]/2
            K[5] = K[5]/2
            
            self.camera_infos[camera_index].append(np.array(K).reshape(3,3))

    def get_camera_intrinsic(self, index):
        if self.camera_infos[index] is None:
            return None
        return np.array(self.camera_infos[index].K).reshape(3, 3)