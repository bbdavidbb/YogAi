import os
import sys
import argparse
import ast
import cv2
import torch
from vidgear.gears import CamGear
import sys
sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict    

def get_keypoint(camera_id = 0, filename = None, hrnet_c = 48, hrnet_j = 17, hrnet_weights = "./weights/pose_hrnet_w48_384x288.pth", hrnet_joints_set = "coco", image_resolution = '(384, 288)', single_person = True,
         max_batch_size = 16, disable_vidgear = False, device = None):
    if torch.cuda.is_available() and True:
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda:0')
    else:
            device = torch.device('cpu')
    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
	
    if filename is not None:
        #video = cv2.VideoCapture(filename)
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        #assert video.isOpened()
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
    pts = model.predict(image)
    resolution = image.shape
    x_len = resolution[0]
    y_len = resolution[1]
    vector = []
    keypoints = pts[0]
    for pt in keypoints:
        pt = list(pt)
        temp = []
        temp.append((pt[0]/x_len))
        temp.append((pt[1]/y_len))
        vector.extend(temp)

    for i, pt in enumerate(pts):
            frame = draw_points_and_skeleton(image, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=i,
                                             points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                             points_palette_samples=10)

    if has_display:
        output_name = filename.split("\\")
        output_name = output_name[-2] + "_" + output_name[-1]			
        cv2.imwrite('tested2\\'+output_name+'.png', frame)
        cv2.imwrite("keypoints_"+filename+".png", frame)
        cv2.imshow('frame.png', frame)
        k = cv2.waitKey(1)
    return vector

def get_file_name(path):
    return os.listdir(path)

if __name__ == '__main__':
    paths = ["yogads\\tree", "yogads\\warrior1", "yogads\\warrior2",
	"yogads\\childs",
	"yogads\\downwarddog",
	"yogads\\plank",
	"yogads\\mountain",
	"yogads\\trianglepose"]
    output_f = open("keypoint_data.txt", "w")
    for path in paths:
        label = path.split("\\")
        label = label[-1]
        files = get_file_name(path)
        for file in files:
            try:
                keypt = get_keypoint(filename = path+"\\"+file)
                output_f.write(label+":"+str(keypt)+"\n")
            except Exception as e:
                print(e)