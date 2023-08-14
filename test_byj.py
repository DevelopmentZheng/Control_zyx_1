
from canny2image_TRT import hackathon

if __name__ == '__main__':
    
    t = hackathon()
   
    print("---------------------Initialize Start!--------------------")
    t.initialize()
    print("--------------------Initialize Finished!------------------")

    print("------------------Transfer ONNX Start!--------------------")
    t.torch2onnx()
    print("----------------Transfer ONNX Finished!-------------------")

