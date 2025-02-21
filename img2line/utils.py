import base64
import numpy as np
from PIL import Image
from io import BytesIO

def encode_to_base64(inp:str):
    """
    path (str): Path to the file to encode
    """

    if isinstance(inp, str):
        with open(inp, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')
    elif isinstance(inp, np.ndarray):
        return base64.b64encode(inp.tobytes()).decode('utf-8')

def decode_base64(base64_str:str):
    buffer = base64.b64decode(base64_str)
    image = Image.open(BytesIO(buffer))
    return np.array(image)