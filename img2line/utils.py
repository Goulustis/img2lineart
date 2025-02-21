import base64
import numpy as np
from PIL import Image
from io import BytesIO

def encode_to_base64(inp, img_format="PNG"):
    """
    Encodes the input to a base64 string.
    
    If inp is a file path (str), it reads the file as binary.
    If inp is a NumPy array, it converts it to a PIL Image and saves it 
    to a BytesIO buffer using the specified image format.
    
    Parameters:
        inp (str or np.ndarray): File path or NumPy array.
        img_format (str): Image format to use when encoding a NumPy array (default "PNG").
        
    Returns:
        str: Base64-encoded string.
    """
    if isinstance(inp, str):
        # Read the file as binary data.
        with open(inp, 'rb') as file:
            return base64.b64encode(file.read()).decode('utf-8')
    elif isinstance(inp, np.ndarray):
        # Convert the NumPy array to a PIL Image.
        image = Image.fromarray(inp)
        # Save the image to a BytesIO buffer in the specified format.
        buffer = BytesIO()
        image.save(buffer, format=img_format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        raise ValueError("Invalid input type")

def decode_base64(base64_str:str):
    buffer = base64.b64decode(base64_str)
    image = Image.open(BytesIO(buffer))
    return np.array(image)

def get_shape(inp):
    if isinstance(inp, str):
        image = Image.open(inp)
        return image.size
    elif isinstance(inp, np.ndarray):
        return inp.shape[:2]
    else:
        raise ValueError("Invalid input type")

def to_binary(image, threshold=165):
    return (image > threshold).astype(np.uint8) * 255