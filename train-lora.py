import os
import sys

# Add the src directory to the Python path if it's not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt

model_type = "sd15"
model_source = "SimianLuo/LCM_Dreamshaper_v7"
output_path = "./engines"
loras = [ ]

for i in range(len(sys.argv)):
    if(i == 0): # app name
        continue
    elif(i == 1): # sd15 / sdxl
        model_type = sys.argv[i]
    elif(i == 2): # model
        model_source = sys.argv[i]
    elif(i == 3): # output name
        output_path = sys.argv[i]
    else:
        loras.append(sys.argv[i])


if(model_type == "sdxl"):
    pipe = StableDiffusionXLPipeline.from_pretrained(model_source).to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_source).to(
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

stream = StreamDiffusion(
    pipe,
    t_index_list=[30,45],
    torch_dtype=torch.float16,
    cfg_type="none", 
)

for lora in loras:
    stream.load_lora(lora)

if(len(loras) > 0):
    stream.fuse_lora(fuse_unet=True, fuse_text_encoder=True, lora_scale=2.5, safe_fusing=False)

if(model_type == "sd15"):
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
else:
    stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl").to(device=pipe.device, dtype=pipe.dtype)

engine_build_options = {
        "opt_batch_size": 2,
        "min_image_resolution": 1024,
        "max_image_resolution": 1024,
        "opt_image_height": 1024,
        "opt_image_width": 1024
}
stream = accelerate_with_tensorrt(
      stream
    , output_path
    , min_batch_size=1
    , max_batch_size=2
    , engine_build_options=engine_build_options
#    , static_shapes=True: only if max_batch_size 1 otherwise clip wants to be doubled
#    use_v2v=True,  # Enable V2V for C++ testing
)

