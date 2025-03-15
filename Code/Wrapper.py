import cv2
import json
import scipy
import torch

import numpy as np

from glob import glob
from ultralytics import YOLO


def Load_Images(Path):
    Images = []
    for file in sorted(glob(Path + '*.png')):
        img = cv2.imread(file)
        Images.append(img)
    return Images

class visual_slam:
    def __init__(self, k_matrix):
        self.k_matrix = k_matrix
        self.detector = cv2.ORB_create(500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.global_rotation = np.eye(3)
        self.global_translation = np.zeros((3, 1))
        self.global_transfomation = np.eye(4)
        self.rotations = []
        self.images = []
        self.descriptors = []
        self.keypoints = []
        self.poses = []
        self.final_points_3d = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO('yolov8n.pt').to(device)
        self.yolo_confidence = 0.3
        
    def Extract_Features(self, img):
        keypoints, descriptors = self.detector.detectAndCompute(img, None)
        keypoints, descriptors = self.remove_keypoints(list(keypoints), list(descriptors))
        return keypoints, descriptors

    def remove_keypoints(self, keypoints, descriptors):
        results = self.model(self.images[-1], stream=True, conf=self.yolo_confidence, verbose=False)
        for result in results:
            boxes = result.boxes.xyxy.to('cpu').numpy().astype(int)
            classes = result.boxes.cls.to('cpu').numpy().astype(int)
        indices = []
        for box in boxes:
            for kp in keypoints:
                if box[0] < kp.pt[0] < box[2] and box[1] < kp.pt[1] < box[3] and classes[0] in [0, 2]:
                    indices.append(keypoints.index(kp))
        indices = list(set(indices))
        for index in sorted(indices, reverse=True):
            del keypoints[index]
            del descriptors[index]
        return tuple(keypoints), np.array(descriptors) if len(descriptors) > 0 else np.array([])

    def Match_Features(self, Descriptors1, Descriptors2):
        matches = self.matcher.match(Descriptors1, Descriptors2)
        matches = sorted(matches, key = lambda x:x.distance)
        return matches

    def estimate_pose(self, kp1, kp2, matches, K):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        return R, t, mask

    def ExtractCameraPose(E):
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        U, S, Vt = scipy.linalg.svd(E)
        R1 = np.dot(np.dot(U, W), Vt)
        R2 = np.dot(np.dot(U, W.T), Vt)
        T1 = U[:, 2]
        T2 = -U[:, 2]
        R = [R1, R1, R2, R2]
        T = [T1, T2, T1, T2]
        for i in range(4):
            if np.linalg.det(R[i]) < 0:
                R[i] = -R[i]
                T[i] = -T[i]
        return R, T

    def triangulate_points(self, kp1, kp2, matches, R, T, r, t, K, mask):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])[mask.ravel() == 1]
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])[mask.ravel() == 1]
        # print(pts1.shape, pts2.shape)
        if pts1.shape[0] < 5 or pts2.shape[0] < 5:
            return [], [], []
        u_pts1 = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
        u_pts2 = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((r, t))
        points_4d = cv2.triangulatePoints(P1, P2, u_pts1, u_pts2)
        points_3d = points_4d[:3] / points_4d[3]
        T_Total = T + np.dot(R, t)
        R_Total = np.dot(r, R)
        points_3d_World = np.dot(R_Total.T, points_3d) + np.tile(T_Total, (points_3d.shape[1]))
        return points_3d_World.T, pts1, pts2


    def run(self, image):
        self.images.append(image)
        if len(self.images) < 2:
            Features1, Descriptors1 = self.Extract_Features(self.images[-1])
            self.descriptors.append(Descriptors1)
            return 0
        print("Processing Frame: ", len(self.images))
        Features1, Descriptors1 = self.Extract_Features(self.images[-2])
        Features2, Descriptors2 = self.Extract_Features(self.images[-1])
        
        Matches = self.Match_Features(Descriptors1, Descriptors2)
        r, t, mask = self.estimate_pose(Features1, Features2, Matches, self.k_matrix)
        Points3D, Features1, Features2 = self.triangulate_points(Features1, Features2, Matches, self.global_rotation, self.global_translation, r, t, self.k_matrix, mask)
        if len(Points3D) == 0:
            self.images.pop()
            return 0
        for Iter in range(len(Points3D)):
            UpdatedPoints3D = Points3D[Iter]
            self.final_points_3d.append(UpdatedPoints3D)

        self.descriptors.append(Descriptors2)
        self.keypoints.append(Features2)
        self.global_translation = self.global_translation + np.dot(self.global_rotation, t)
        self.global_rotation = np.dot(r, self.global_rotation)
        self.poses.append(self.global_translation.tolist())
        self.rotations.append(self.global_rotation.tolist())

        


def main():
    Path = "data/"
    print("Reading Images...")
    Images = Load_Images(Path)
    
    K = np.array([[707.0912, 0, 601.8873],
        [0, 707.0912, 183.1104],
        [0, 0, 1]])

    visual_slam_obj = visual_slam(K)
    for i in range(len(Images)):
        visual_slam_obj.run(Images[i])
        
    F = open("Poses_temp.json", "w")
    Dict = {}
    for i in range(len(visual_slam_obj.poses)):
        Dict[i] = [visual_slam_obj.poses[i][0][0], visual_slam_obj.poses[i][1][0], visual_slam_obj.poses[i][2][0]]
    json.dump(Dict, indent=4, fp=F)
    F.close()
    F = open("Points3D_temp.json", "w")
    Dict = {}
    for i in range(len(visual_slam_obj.final_points_3d)):
        Dict[i] = visual_slam_obj.final_points_3d[i].tolist()
    json.dump(Dict, indent=4, fp=F)
    F.close()

    # return 0

if __name__ == "__main__":
    main()

