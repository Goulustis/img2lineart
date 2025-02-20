# Img2lineart
turns images into line art

## Installation
- install [stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) with this commit: 82a973c04367123ae98bd9abdf80d9eda9b910e2
- download control net and diffusion checkpoint:
    - [dreamshaper_8.safetensors](https://civitai.com/models/4384/dreamshaper)
    - [control_canny-fp16](https://huggingface.co/webui/ControlNet-modules-safetensors/tree/main)
- start diffusion server with
```bash
# NOTE: remove --nowebui for gui; need to update the url if removed
bash webui.sh --api --nowebui
```

## Usage
do this after done todos

## Todo
- [ ] change the output dir, looks ugly right now
- [ ] finish implementing the pipeline
    - do ori_img -> pipeline -> out_1 -> pipeline -> out2
