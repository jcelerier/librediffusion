# librediffusion

A C++ / CUDA / TensorRT implementation of StreamDiffusion

Implemented in [ossia score](https://ossia.io)

On a RTX 5090 at 1 step: 

SDXL Turbo 1024x1024: stable 26 fps
![sdxl](https://github.com/user-attachments/assets/340bf804-6822-46c6-87d8-bf784722f3b5)

SD Turbo 512x512: stable 96 fps
![sdturbo](https://github.com/user-attachments/assets/10ebc9f7-d5b7-487b-a301-b1b7e2370d55)

SDXS: above 600 fps
![sdxs](https://github.com/user-attachments/assets/9f735f86-d162-4c4c-b781-163cadf166a5)

Models need to be converted to TensorRT through the Python script [train-lora.py] beforehand:

```bash
$ uv run train-lora.py --model stabilityai/sd-turbo --min-batch 1 --max-batch 1 --opt-batch 1 --min-resolution 512 --max-resolution 1024 --output ./engines-sd-turbo
```
