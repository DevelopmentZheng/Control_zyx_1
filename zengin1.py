import tensorrt as trt
from cuda import cudart
import numpy as np
import torch

class ClipEngine():
    def __init__(self):
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        
        with open("./clip.plan", 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.clip_to_engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.clip_to_engine.create_execution_context()
        #print(f"{self.clip_to_engine} is  ok" )
    def clip_enging(self,prompt):
        engine = self.clip_to_engine
        # print(f"获取输入prompt{prompt.shape}  开始推理clip的tranformer")
        # 输入出张量量
        nIO = engine.num_io_tensors
        #名字
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
       
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
        #创建流
        
        #
        self.context.set_input_shape(lTensorName[0], [1,  77])
        # for i in range(nIO):
        #     print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), self.context.get_tensor_shape(lTensorName[i]), lTensorName[i])
        # #创建cpu内存
        bufferH = []
        bufferH.append(np.ascontiguousarray(prompt))
        #输出
        for i in range(nInput, nIO):
            bufferH.append(np.empty(self.context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        for i in range(nIO):
            self.context.set_tensor_address(lTensorName[i], int(bufferD[i]))
        
        self.context.execute_async_v3(0)
        for i in range(nInput, nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            #print(f" bufferH   {i}  {bufferH[i].shape}  {bufferH[i].dtype} ")
        for b in bufferD:
            cudart.cudaFree(b)
    
        #print(f" 推理结束clipout ")
        return  torch.tensor(bufferH[1]).to("cuda")
        
    
 
class UnetEngine():
    def __init__(self):
        TRT_LOGGER_unet = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER_unet, '')
        with open("./cu.plan", 'rb') as sd, trt.Runtime(TRT_LOGGER_unet) as unet_runtime:
            self.sd_control_engine = unet_runtime.deserialize_cuda_engine(sd.read()) 
        
         #创建流
        self.context  = self.sd_control_engine.create_execution_context()
        # print(f"cu{ self.sd_control_engine} is ok ")
    def unet_enging(self, x_noisy ,timestestep_in, cond_c_crossattn ,cond_c_concat):
        # print(f"获取输入 开始推理unet")
        # 输入出张量量
        engine = self.sd_control_engine
        nIO = engine.num_io_tensors
        #名字
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
        
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
       
        #

        self.context.set_binding_shape(0, (1, 4, 32, 48))
        self.context.set_binding_shape(1, (1,))
        self.context.set_binding_shape(2, (1, 3, 256, 384))
        self.context.set_binding_shape(3, (1, 77, 768))
        
        # for i in range(nIO):
        #     print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), self.context.get_tensor_shape(lTensorName[i]), lTensorName[i])
        #创建cpu内存
        bufferH = []
        #输入
        bufferH.append(x_noisy.cpu().numpy())
        bufferH.append(timestestep_in.cpu().numpy())
        bufferH.append(cond_c_crossattn.cpu().numpy())
        bufferH.append(cond_c_concat.cpu().numpy())
        #输出分配缓存
        for i in range(nInput, nIO):
            bufferH.append(np.empty(self.context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
        #
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        # 复制数据
        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        
        for i in range(nIO):
            self.context.set_tensor_address(lTensorName[i], int(bufferD[i]))
        
        self.context.execute_async_v3(0)
        
        for i in range(nInput, nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            # print(f" {i}  {bufferH[i].shape}  ")
        
        for b in bufferD:
            cudart.cudaFree(b)
        
        #print(f" 推理结束unetconout")
        return  torch.tensor(bufferH[4]).to("cuda")
        
# TRT_LOGGER_vae = trt.Logger(trt.Logger.INFO)
# with open("./vae.engine", 'rb') as vae, trt.Runtime(TRT_LOGGER_vae) as vae_runtime:
#     vae_to_engine = vae_runtime.deserialize_cuda_engine(vae.read()) 
#     if sd_control_engine is not None:
#        print(" vae_to_engine ok")
    
class VaeEngine():
    def __init__(self):
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        with open("./vae.plan", 'rb') as self.f, trt.Runtime(TRT_LOGGER ) as self.runtime:
            self.vae_engine = self.runtime.deserialize_cuda_engine(self.f.read())
            #print(f"{self.vae_engine} is ok ")
        self.context = self.vae_engine.create_execution_context()
    def vae_enging(self,img):
        engine = self.vae_engine
        # print(f"获取输入img{img.shape}  开始推理vae")
        # 输入出张量量
        nIO = engine.num_io_tensors
        #名字
        lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    
        nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)
        #创建流
      
        #
        self.context.set_input_shape(lTensorName[0], [1,4,32,48])
    
        # for i in range(nIO):
        #     print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), self.context.get_tensor_shape(lTensorName[i]), lTensorName[i])
        #创建cpu内存
        bufferH = []
        
        bufferH.append(np.ascontiguousarray(img))
        #输出
        for i in range(nInput, nIO):
            bufferH.append(np.empty(self.context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
        
        bufferD = []
        for i in range(nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
        
        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        
        for i in range(nIO):
            self.context.set_tensor_address(lTensorName[i], int(bufferD[i]))
        
        self.context.execute_async_v3(0)
        
        for i in range(nInput, nIO):
            cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            #print(f" {i}  {bufferH[i].shape}")
        
        for b in bufferD:
            cudart.cudaFree(b)
        
        #print(f" 推理结束vae out")
        return  bufferH[1]
    
    
    
