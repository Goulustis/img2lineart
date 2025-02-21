from img2line.sd_api import SDAPI, SDAPI_config
from PIL import Image
import numpy as np

def sd_lineart():
    img_f = "path_to_your_image.jpeg"
    img = np.array(Image.open(img_f))
    sd: SDAPI = SDAPI_config(webui_server_url="http://127.0.0.1:7860").setup()

    out = sd.to_lineart(img, out_sz=512, seed=32)
    Image.fromarray(out).save("final.png")

if __name__ == '__main__':
    sd_lineart()

