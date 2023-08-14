from share import *
import config
import os
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import tensorrt as trt

device = torch.device('cuda') 

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from zengin1 import ClipEngine ,VaeEngine,UnetEngine
###

class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.clip_engine =None
        self.vae_engine =None
        self.unet_controlnet_engine =None
        # self.tokenizer =None
        self.clip_engine = ClipEngine()
        self.vaeEngine = VaeEngine()

        clip_model = getattr(self.model, "cond_stage_model")
        self.tokenizer = clip_model.tokenizer

        self.state_dict = {
            "clip": "cond_stage_model",
            "control_net": "control_model",
            "unet": "diffusion_model",
            "vae": "first_stage_model",
            "Control_Unet":""
        }
        self.onnx_path_dict = {
            "clip": "./clip.onnx",
            "control_net": "./control_net.onnx",
            "unet": "./unet_new.onnx",
            "vae": "./vae_new.onnx",
            "Control_Unet":"./onnx_model/Control_Unet_new.onnx"
        }
       

    def torch2onnx(self):
        for k, v in self.state_dict.items():
            if k != "unet":
                if k == "Control_Unet":
                    temp_model = self.model
                else:
                    temp_model = getattr(self.model, v)
            else:
                temp_model = getattr(self.model.model, v)
            if k == "clip":
                model = temp_model.transformer
                
                self.tokenizer = temp_model.tokenizer
                if os.path.exists(self.onnx_path_dict[k]):
                    print('Clip.onnx already exists!')
                    continue
                inputs = torch.ones((1,77),dtype=torch.int64).cuda()
                input_names = ["input_ids"]
                output_names = ["text_embedding"]
                dynamic_axes = {"input_ids":[0]}

                
                torch.onnx.export(
                    model,
                    inputs,
                    self.onnx_path_dict[k],
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )
                print("vae.onnx export ")
                os.system("trtexec --onnx=./clip.onnx --saveEngine=clip.plan --optShapes=input_ids:1x77")

            elif k == "vae":

                if os.path.exists(self.onnx_path_dict[k]):
                    print('Vae.onnx already exists!')
                    continue
                inputs = torch.randn((1,4,32,48)).cuda()

                input_names = ["embedding_input"]
                output_names = ["pred"]
                dynamic_axes = {"embedding_input":[0]}

                vae_model = temp_model

                torch.onnx.export(
                    vae_model,
                    inputs,
                    self.onnx_path_dict[k],
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )
                print("vae.onnx export ")
                os.system("trtexec --onnx=./vae_new.onnx --saveEngine=vae.plan --optShapes=embedding_input:1x4x32x48")
                   
            elif k == "Control_Unet":

                if os.path.exists(self.onnx_path_dict[k]):
                    print('Control_Unet.onnx already exists!')
                    continue
                x_noisy = torch.randn((1, 4, 32, 48), dtype=torch.float32).to("cuda")
                timestestep_in= torch.tensor([1], dtype=torch.int64).to("cuda")
                cond_c_crossattn = torch.randn((1, 77, 768), dtype=torch.float32).to("cuda")
                cond_c_concat = torch.randn((1, 3, 256, 384), dtype=torch.float32).to("cuda")

                input_names = ["x_noisy", "timestestep_in", "cond_c_concat", "cond_c_crossattn"]
                output_names = ["pred"]
                dynamic_axes = {"x_noisy":[0],
                                "cond_c_concat":[0],
                                "cond_c_crossattn":[0]}

                torch.onnx.export(
                    self.model,
                    (x_noisy,timestestep_in, cond_c_concat, cond_c_crossattn),#cond_txt = cond_c_crossattn
                    self.onnx_path_dict[k],
                    export_params=True,
                    opset_version=16,
                    do_constant_folding=True,
                    keep_initializers_as_inputs=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )

                print(" export Control_unet_model.onnx is ok ")
                os.system("trtexec --onnx=./onnx_model/Control_Unet_new.onnx --saveEngine=cu.plan --optShapes=x_noisy:1x4x32x48,timestestep_in:1,cond_c_crossattn:1x77x768,cond_c_concat:1x3x256x384")
                
    # prompt 好的， a_prompt补充的  n_prompt差的
    def process_trt(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():


            # =======================step 1======================
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)
            # =======================step 1======================


            # -======================step 2====================
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

           
            # 条件prompt
            #cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            # 负面prompt
            #un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)
            # condinfo = self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)
            # print(f"cond----{condinfo.shape}")
            # print(f"cond----{condinfo.device}")
            # print(f"cond----{condinfo.dtype}")
            
            
            # 修改的
            
            batch_encoding = self.tokenizer([prompt + ', ' + a_prompt] * num_samples, truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"]
            cond1 = self.clip_engine.clip_enging(tokens.numpy())
            cond_change = {"c_concat": [control], "c_crossattn": [cond1.to(device)]}
            
            batch_encoding = self.tokenizer([n_prompt] * num_samples, truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"]
            un_cond1 = self.clip_engine.clip_enging(tokens.numpy())
            un_cond_change = {"c_concat": None if guess_mode else [control], "c_crossattn": [un_cond1]}
            
            # print(f"cond.shape{cond}")
            # print(f"un_cond.shape{un_cond.shape}")
            

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)
            # -======================step 2====================




            # controlnet权重    不需要
            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            

            # =====================step 3==============================
            samples, intermediates = self.ddim_sampler._trt_sample(ddim_steps, num_samples,
                                                        shape, cond_change, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond_change)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)
            # =====================step 3==============================


            #   ======================step 4=====================
            #x_samples = self.model.decode_first_stage(samples) # out torch.float32   torch.Size([1, 3, 256, 384]) 
            #print("----x_samples------ ")
            #print(f"x_samples1  {x_samples.shape} ")
            #x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            
            #change
            samples = 1. / 0.18215 * samples
            x_samples = self.vaeEngine.vae_enging(samples.cpu().numpy())
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
            #   ======================step 4=====================

            
        return results


    def process_torch(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            # np.save('./data/control.npy', control.detach().cpu().numpy())
            # np.save('./data/prompt.npy', self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples).detach().cpu().numpy())
            # np.save('./data/n_prompt.npy', self.model.get_learned_conditioning([n_prompt] * num_samples).detach().cpu().numpy())

            # 条件prompt
            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            # 负面prompt
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            # controlnet权重
            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            
            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return results