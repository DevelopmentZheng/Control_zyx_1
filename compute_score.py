import numpy as np
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import cv2
import datetime
from canny2image_TRT import hackathon

import torch

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx]).to("cuda")

def PD(base_img, new_img):
    inception_feature_ref, _ = fid_score.calculate_activation_statistics([base_img], model, batch_size = 1, device="cuda")
    inception_feature, _ = fid_score.calculate_activation_statistics([new_img], model, batch_size = 1, device="cuda")
    pd_value = np.linalg.norm(inception_feature - inception_feature_ref)
    pd_string = F"Perceptual distance to: {pd_value:.2f}"
    print(pd_string)
    return pd_value

scores = []
latencys = []
hk = hackathon()
hk.initialize()

#hk.torch2onnx()
#20
for i in range(20):
    path = "/home/player/pictures_croped/bird_"+ str(i) + ".jpg"
    img = cv2.imread(path)
    start = datetime.datetime.now().timestamp()
    new_img = hk.process_trt(img,
            "a bird", 
            "best quality, extremely detailed", 
            "longbody, lowres, bad anatomy, bad hands, missing fingers", 
            1, 
            # 
            256, 
            # 20步
            10,
            False, 
            1, 
            9, 
            2946901, 
            0.0, 
            100, 
            200)
    end = datetime.datetime.now().timestamp()
    print("time cost is: ", (end-start)*1000)
    new_path = "./bird_"+ str(i) + ".jpg"
    cv2.imwrite(new_path, new_img[0])

    torch.cuda.empty_cache()
    # generate the base_img by running the pytorch fp32 pipeline (origin code in canny2image_TRT.py)


    start1 = datetime.datetime.now().timestamp()
    new_img1 = hk.process_torch(img,
            "a bird", 
            "best quality, extremely detailed", 
            "longbody, lowres, bad anatomy, bad hands, missing fingers", 
            1, 
            # 
            256, 
            # 20步
            20,
            False, 
            1, 
            9, 
            2946901, 
            0.0, 
            100, 
            200)
    end1 = datetime.datetime.now().timestamp()
    print("time1 cost is: ", (end1-start1)*1000)
    base_path =  "./base_img"+ str(i) + ".jpg" 
    cv2.imwrite(base_path, new_img1[0])
    
    score = PD(base_path, new_path)
    print("score is: ", score)

