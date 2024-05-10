import base64
import mimetypes
from math import ceil

import streamlit as st
from PIL import Image


# https://community.openai.com/t/how-do-i-calculate-image-tokens-in-gpt4-vision/492318
def calculate_image_tokens(width: int, height: int):
    if width > 2048 or height > 2048:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            width, height = 2048, int(2048 / aspect_ratio)
        else:
            width, height = int(2048 * aspect_ratio), 2048

    if width >= height > 768:
        width, height = int((768 / height) * width), 768
    elif height > width > 768:
        width, height = 768, int((768 / width) * height)

    tiles_width = ceil(width / 512)
    tiles_height = ceil(height / 512)
    total_tokens = 85 + 170 * (tiles_width * tiles_height)

    return total_tokens


# streamlit specific
def encode_image_with_mime_type(image_file):
    mime_type, _ = mimetypes.guess_type(image_file.name)
    if mime_type is None:
        raise ValueError("Could not determine the MIME type of the image")
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_image}"


def encode_image_path_with_mime_type(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError("Could not determine the MIME type of the image")
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_image}"


def get_image_dimensions(uploaded_file):
    # Load the image using PIL
    image = Image.open(uploaded_file)
    # Extract width and height
    width, height = image.size
    st.write(f"Uploaded image dimensions: {width} x {height}")
    return width, height
