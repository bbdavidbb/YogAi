import os
import sys
import argparse
import ast
import cv2
import torch
import glob
import time
import pickle
from sklearn.neural_network import MLPClassifier
from vidgear.gears import CamGear
sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,100)
fontScale              = 1
fontColor              = (0,0,0)
lineType               = 2


def open_app(camera_id = 0, filename = None, hrnet_c = 48, hrnet_j = 17, hrnet_weights = "./weights/pose_hrnet_w48_384x288.pth", hrnet_joints_set = "coco", image_resolution = '(384, 288)', single_person = True,max_batch_size = 16, disable_vidgear = False, device = None):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available() and True:
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    if filename is not None:
        video = cv2.VideoCapture(filename)
        assert video.isOpened()
    else:
        if disable_vidgear:
            video = cv2.VideoCapture(camera_id)
            assert video.isOpened()
        else:
            video = CamGear(camera_id).start()

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        resolution=image_resolution,
        multiperson=not single_person,
        max_batch_size=max_batch_size,
        device=device
    )
    loaded_model = pickle.load(open("mlp_model_best.sav", 'rb'))
    no_to_label = {0:"tree", 1:"warrior1", 2:"warrior2", 3:"childs",4:"downwarddog",5:"plank",6:"mountain",7:"trianglepose"}
    image_to_blob = {}
    for id,path in no_to_label.items():
        images = [cv2.imread(file) for file in glob.glob('sampleposes\\'+path+'.jpg')]
        image_to_blob[id] = images
    while True:
        if filename is not None or disable_vidgear:
            ret, frame = video.read()
            if not ret:
                break
        else:
            frame = video.read()
            if frame is None:
                break
        pts = model.predict(frame)
        resolution = frame.shape
        x_len = resolution[0]
        y_len = resolution[1]
        vector = []
        if len(pts) == 0:
            continue
        keypoints = pts[0]

        for pt in keypoints:
            pt = list(pt)
            temp = []
            temp.append((pt[0]/x_len))
            temp.append((pt[1]/y_len))
            vector.extend(temp)

        vector = list(vector)
        predicted_pose = loaded_model.predict([vector]) 
        text = no_to_label[predicted_pose[0]] + " pose"
        cv2.putText(image_to_blob[predicted_pose[0]][0], text, bottomLeftCornerOfText, font, fontScale,fontColor,lineType) 
        cv2.imshow("Suggestion",image_to_blob[predicted_pose[0]][0])
        k= cv2.waitKey(1)
        for i, pt in enumerate(pts):
            frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=i,
                                             points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                             points_palette_samples=10)

        if has_display:
            cv2.imshow('frame.png', frame)
            k = cv2.waitKey(1)
            if k == 27:  # Esc button
                if disable_vidgear:
                    video.release()
                else:
                    video.stop()
                break
        else:
            cv2.imwrite('frame.png', frame)

if __name__ == '__main__':
    open_app()
