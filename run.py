from datetime import datetime
import urllib.request
import base64
import json
import time
import os

# webui_server_url = 'http://127.0.0.1:7860'
webui_server_url = 'http://0.0.0.0:7861'

out_dir = 'api_out'
out_dir_t2i = os.path.join(out_dir, 'txt2img')
out_dir_i2i = os.path.join(out_dir, 'img2img')
os.makedirs(out_dir_t2i, exist_ok=True)
os.makedirs(out_dir_i2i, exist_ok=True)


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))


def call_img2img_api(**payload):
    response = call_api('sdapi/v1/img2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_i2i, f'img2img-{timestamp()}-{index}.png')
        decode_and_save_base64(image, save_path)


if __name__ == '__main__':

    init_img_f = "/home/boss/Downloads/blank.jpeg"
    control_net_img_f = "/home/boss/Downloads/image.png"

    init_images = [
        encode_file_to_base64(init_img_f),
    ]

    batch_size = 1
    payload = {
        "prompt": "line art drawing, line art, black line art, black line, black color, black lines, a line drawing",
        "seed": 32,
        "steps": 20,
        "width": 512,
        "height": 512,
        "denoising_strength": 0.93,
        "sampler_name": "DPM++ 3M SDE",
        "n_iter": 1,
        "init_images": init_images,
        "batch_size": batch_size if len(init_images) == 1 else len(init_images),
        "override_settings": {
            'sd_model_checkpoint': "dreamshaper_8.safetensors [879db523c3]",  # this can use to switch sd model
        },
        "alwayson_scripts": {
            "ControlNet": {
                "args": [
                    {
                        "batch_images": "",
                        "control_mode": "Balanced",
                        "enabled": True,
                        "guidance_end": 1,
                        "guidance_start": 0,
                        "image": {
                            "image": encode_file_to_base64(control_net_img_f),
                            "mask": None  # base64, None when not need
                        },
                        "input_mode": "simple",
                        "is_ui": True,
                        "loopback": False,
                        "low_vram": False,
                        "model": "control_canny-fp16 [e3fe7712]",
                        "module": "canny",
                        "output_dir": "",
                        "pixel_perfect": True,
                        "processor_res": 512,
                        # "resize_mode": "Crop and Resize",
                        "resize_mode": "Just Resize",
                        "threshold_a": 100,
                        "threshold_b": 200,
                        "weight": 1
                    }
                ]
            }
        }
        
    }
    call_img2img_api(**payload)

# start server with:
# bash webui.sh --api --nowebui