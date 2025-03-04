from dataclasses import dataclass, field
from typing import Type
import json
import urllib.request
import random
from loguru import logger
import numpy as np

from img2line.config import InstantiateConfig
from img2line.utils import encode_to_base64, decode_base64, get_shape


@dataclass
class SDAPI_config(InstantiateConfig):
    _target: Type = field(default_factory = lambda : SDAPI)
    
    webui_server_url: str = 'http://0.0.0.0:7861'
    """ URL of the webui server """


class SDAPI:
    def __init__(self, config: SDAPI_config):
        self.config = config
        self.server_url = config.webui_server_url
    

    def img2img(self, inp:str | np.ndarray, ctrl_inp:str | np.ndarray, out_sz:int=None, **payload_override):
        """
        inp_f: str | numpy array
            Path to the input image file
        ctrl_inp_f: str | numpy array
            Path to control net image file
        """

        h, w = get_shape(inp) if out_sz is None else (out_sz, out_sz)

        inp_img = encode_to_base64(inp)
        ctrl_img = encode_to_base64(ctrl_inp)

        init_images = [inp_img]
        batch_size = 1 # number of images to generate per input
        seed = random.randint(0, 1000000) if payload_override.get("seed") is None else payload_override["seed"]

        payload = {
                "prompt": "line art drawing, line art, black line art, black line, black color, black lines, a line drawing, monochrome",
                "seed": seed,
                "steps": 20,
                "width": w,
                "height": h,
                "denoising_strength": 0.93,
                "sampler_name": "DPM++ 3M SDE",
                "n_iter": 1,
                "init_images": init_images,
                "batch_size": batch_size,
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
                                    "image": ctrl_img,
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
                                "resize_mode": "Crop and Resize",
                                # "resize_mode": "Just Resize",
                                "threshold_a": 100,
                                "threshold_b": 200,
                                "weight": 1
                            }
                        ]
                    }
                }
                
            }
        payload.update(payload_override)

        logger.info(f"generating image with seed {seed}")
        
        api_endpoint = 'sdapi/v1/img2img'
        response = self._call_api(api_endpoint, **payload)
        imgs = response["images"]  
        img = decode_base64(imgs[0]) # NOTE: batch_size = 1
        return img

    
    def _call_api(self, api_endpoint, **payload):
        data = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            f'{self.server_url}/{api_endpoint}',
            headers={'Content-Type': 'application/json'},
            data=data,
        )
        response = urllib.request.urlopen(request)
        return json.loads(response.read().decode('utf-8'))
    
    def to_lineart(self, inp:np.ndarray, out_sz:int=None, n_iter:int=2, **payload_override):
        """
        inp: np.ndarray
            Input image; expect [0 - 255] range
        out_sz: int
            Size of the output image
        n_iter: int
            Number of iterations to run the model
        """
        img = inp
        for _ in range(n_iter):
            blank = img.copy()
            blank.fill(255)
            img = self.img2img(blank, img, out_sz=out_sz, **payload_override)
        return img