# Given camera params, create observation

import numpy as np
import cv2


class camera:
    def __init__(self, ref_point_info):
        self.img_width = float(ref_point_info["EXIF"]["ExifImageWidth"])
        self.img_height = float(ref_point_info["EXIF"]["ExifImageHeight"])
        self.cx = self.img_width / 2
        self.cy = self.img_height / 2

        self.f = float(ref_point_info["EXIF"]["FocalLength"])
        self.pixel_size = 3.3e-3  # e-6, to have width in mm
        self.sensor_width, self.sensor_height = (
            self.img_width * self.pixel_size,
            self.img_height * self.pixel_size,
        )
        self.f_x = (self.f * self.img_width) / self.sensor_width
        self.f_y = (self.f * self.img_height) / self.sensor_height
        self.alt_offset = float(ref_point_info["XMPInfo"]["RelativeAltitude"])
        self.K = [
            [self.f_x, 0, self.cx],
            [0, self.f_y, self.cy],
            [0, 0, 1],
        ]  # instrinsic matrix

    def camera_characteristics(self):
        print(
            f"img size: w{self.img_width}-h{self.img_height}\n focal length: {self.f} fx:{self.f_x} fy:{self.f_y}\n sensor size: w{self.sensor_width}-h{self.sensor_height}\n"
        )

    def _get_image_corners(self):
        img_corners = np.array(
            [
                [0, 0],
                [self.img_width, 0],
                [0, self.img_height],
                [self.img_width, self.img_height],
            ]
        )
        return img_corners

    def _normalize_corners(self, img_corners=None):
        if img_corners == None:
            img_corners = self._get_image_corners()
        img_corners_normalized = np.array(
            [
                [(x - self.cx) / self.f_x, (y - self.cy) / self.f_y, 1.0]
                for x, y in img_corners
            ]
        )
        return img_corners_normalized

    def imgToWorldCoord(self, T, point_info):
        # Transform rays into world coordinates
        T[2] += self.alt_offset

        # Extract drone flight orientation
        flight_yaw, flight_pitch, flight_roll = (
            np.deg2rad(float(point_info["XMPInfo"]["FlightYawDegree"])),
            np.deg2rad(float(point_info["XMPInfo"]["FlightPitchDegree"])),
            np.deg2rad(float(point_info["XMPInfo"]["FlightRollDegree"])),
        )
        # print(f"Drone flight angles (deg): yaw={np.rad2deg(flight_yaw)}, pitch={np.rad2deg(flight_pitch)}, roll={np.rad2deg(flight_roll)}")

        # Extract gimbal orientation
        gimbal_yaw, gimbal_pitch, gimbal_roll = (
            np.deg2rad(float(point_info["XMPInfo"]["GimbalYawDegree"])),
            np.deg2rad(float(point_info["XMPInfo"]["GimbalPitchDegree"]) + 90),
            np.deg2rad(float(point_info["XMPInfo"]["GimbalRollDegree"])),
            # 0.0,
        )
        # print(f"Gimbal angles (deg): yaw={np.rad2deg(gimbal_yaw)}, pitch={np.rad2deg(gimbal_pitch)}, roll={np.rad2deg(gimbal_roll)}")

        # Calculate combined rotation (world to camera)
        R_flight = (
            cv2.Rodrigues(np.array([0, 0, flight_yaw]))[0]
            @ cv2.Rodrigues(np.array([0, flight_pitch, 0]))[0]
            @ cv2.Rodrigues(np.array([flight_roll, 0, 0]))[0]
        )
        R_gimbal = (
            cv2.Rodrigues(np.array([0, 0, gimbal_yaw]))[0]
            @ cv2.Rodrigues(np.array([0, gimbal_pitch, 0]))[0]
            @ cv2.Rodrigues(np.array([gimbal_roll, 0, 0]))[0]
        )

        R_world_to_camera = R_flight @ R_gimbal

        # # https://developer.dji.com/mobile-sdk/documentation/introduction/component-guide-gimbal.html

        # print(f"gimbal: y {np.rad2deg(yaw)}; p {np.rad2deg(pitch)}; r {np.rad2deg(roll)}")
        world_corners = []
        altitude = abs(T[2])  # Drone height above ground

        # Normalize corners in the camera frame
        fov = np.deg2rad(60)
        tan = np.tan(fov / 2)
        c_top_left = [-tan, tan, -1]
        c_top_right = [tan, tan, -1]
        c_bottom_left = [-tan, -tan, -1]
        c_bottom_right = [tan, -tan, -1]
        corners_camera_frame = (
            np.array([c_top_left, c_top_right, c_bottom_right, c_bottom_left])
            * altitude
        )

        # img_corners_normalized = self._normalize_corners()

        # Scale normalized corners to actual image size in meters
        # corners_camera_frame = [
        #     [corner[0] * altitude, corner[1] * altitude, -altitude]
        #     for corner in img_corners_normalized
        # ]
        for corner in corners_camera_frame:
            corner_world = R_world_to_camera @ np.array(corner)  # Rotate to world frame
            corner_world += T  # Translate to drone position
            world_corners.append(corner_world)

        return world_corners

    def get_fov_corners_in_ned(self, T, point_info):
        T[2] += self.alt_offset
        # img_corners_normalized = (
        #     self._normalize_corners()
        # )  # Assumes corners are defined in image plane coordinates
        # aspect_ratio = self.img_width / self.img_height
        corners = self._get_image_corners()
        corners = np.array([[c[0], c[1], 1] for c in corners])
        yaw, pitch, roll = (
            np.deg2rad(float(point_info["XMPInfo"]["FlightYawDegree"])),
            np.deg2rad(float(point_info["XMPInfo"]["FlightPitchDegree"])),
            np.deg2rad(float(point_info["XMPInfo"]["FlightRollDegree"])),
        )
        R_z = cv2.Rodrigues(np.array([0, 0, yaw]))[0]
        R_y = cv2.Rodrigues(np.array([0, pitch, 0]))[0]
        R_x = cv2.Rodrigues(np.array([roll, 0, 0]))[0]

        R_drone = R_x @ R_y @ R_z
        print(
            f" flight yaw:{np.rad2deg(yaw)}; pitch: {np.rad2deg(pitch)}; roll:{np.rad2deg(roll)}"
        )

        yaw, pitch, roll = (
            np.deg2rad(float(point_info["XMPInfo"]["GimbalYawDegree"])),
            np.deg2rad(float(point_info["XMPInfo"]["GimbalPitchDegree"])),
            np.deg2rad(float(point_info["XMPInfo"]["GimbalRollDegree"])),
        )
        print(
            f"yaw:{np.rad2deg(yaw)}; ptch: {np.rad2deg(pitch)}; roll:{np.rad2deg(roll)}"
        )

        R_z = cv2.Rodrigues(np.array([0, 0, yaw]))[0]
        R_y = cv2.Rodrigues(np.array([0, pitch, 0]))[0]
        R_x = cv2.Rodrigues(np.array([roll, 0, 0]))[0]

        R_gimbal = R_x @ R_y @ R_z
        R = R_drone
        inv_R = np.linalg.inv(R)
        inv_K = np.linalg.inv(self.K)
        # print(self.K)
        # print(inv_K)
        fov_corners_ned = []
        for corner in corners:
            # Rotate the normalized corner vector to the NED frame
            corner_ned = inv_R @ (inv_K @ corner)
            print(f" corner- {T[2]*inv_K @ corner }")
            print(f" corner-ned {T[2]*inv_K @ corner }")
            print(f" T {T}")
            fov_corners_ned.append(corner_ned)

        return fov_corners_ned

    def forwardProj(self, T, point_info):
        yaw, pitch, roll = (
            np.deg2rad(float(point_info["XMPInfo"]["GimbalYawDegree"])),
            np.deg2rad(float(point_info["XMPInfo"]["GimbalPitchDegree"])),
            np.deg2rad(float(point_info["XMPInfo"]["GimbalRollDegree"])),
        )
        R_z = cv2.Rodrigues(np.array([0, 0, yaw]))[0]
        R_y = cv2.Rodrigues(np.array([0, pitch, 0]))[0]
        R_x = cv2.Rodrigues(np.array([roll, 0, 0]))[0]
        R = R_z @ R_y @ R_x  # Combine rotations in ZYX order
        T[2] += self.alt_offset
        inv_k = np.linalg.inv(self.K)
        H_c2v = np.dot(np.dot(self.K, R), inv_k)
