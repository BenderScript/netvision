import asyncio
import mimetypes
import os
from io import BytesIO
from math import ceil

import openai
import streamlit as st
import base64
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image

# Set your OpenAI API key here
load_dotenv(override=True)
st.session_state['openai_client'] = AsyncOpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = st.session_state['openai_client']


def encode_image_with_mime_type(image_file):
    mime_type, _ = mimetypes.guess_type(image_file.name)
    if mime_type is None:
        raise ValueError("Could not determine the MIME type of the image")
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_image}"


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


# https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format
# finish_reason="length"
async def chat_with_model(session_messages):
    try:
        with st.spinner('Working on your prompt, please wait...'):
            response = await client.chat.completions.create(model="gpt-4-turbo", messages=session_messages)
            message_response = response.choices[0]
            content = message_response.message.content
            finish_reason = message_response.finish_reason

            if finish_reason == 'length':
                print("Response stopped because the maximum response length was reached.")
            elif finish_reason == 'stop':
                print("Response stopped due to reaching a stop sequence.")

        return content
    except openai.error.InvalidRequestError as e:
        if "tokens" in str(e):
            print("Error: The prompt is too large for the context or the response limit was exceeded.")
        else:
            print(f"An unexpected InvalidRequestError occurred: {e}")
    except openai.error.OpenAIError as e:
        print(f"OpenAI Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


async def chat_with_model_vision(session_messages):
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": session_messages[-1]["content"]
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": st.session_state['image_data']
                    }
                }
            ]
        }
    ]
    with st.spinner('Working on your prompt, please wait...'):
        response = await client.chat.completions.create(model="gpt-4-turbo", messages=session_messages)
        return response.choices[0].message.content


def display_messages():
    if 'messages' in st.session_state:
        for msg in st.session_state['messages']:
            with st.chat_message(msg['role']):
                if isinstance(msg['content'], list):  # Check if content is a list of items
                    for content in msg['content']:
                        if content['type'] == 'text':
                            st.markdown(content['text'])
                        elif content['type'] == 'image_url':
                            # Assuming the image data comes directly in a base64 format
                            base64_image = content['image_url']['url']
                            header, encoded = base64_image.split(",", 1)

                            # Decode the image data
                            data = base64.b64decode(encoded)

                            # Create a PIL Image instance
                            image = Image.open(BytesIO(data))
                            st.image(image, caption='Uploaded Image', use_column_width=True)
                else:  # Handle the case where content is a simple string
                    st.markdown(msg['content'])


def append_message(role, content):
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    st.session_state['messages'].append({"role": role, "content": content})


def append_message_list(messages):
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    st.session_state['messages'].extend(messages)


async def handle_image_upload(uploaded_file):
    """Process and display the uploaded image."""
    data_uri = encode_image_with_mime_type(uploaded_file)
    st.session_state['image_data'] = data_uri  # Store the encoded image in session state
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is an image of a network diagram using Cisco products. Please explain in detail "
                            "the topology including products used and if applicable: VLANs, OSPF routing areas, "
                            "BGP ASNs, and any other relevant networking information to understand the data flow."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_uri
                    }
                }
            ]
        }
    ]
    append_message_list(PROMPT_MESSAGES)
    assistant_response = await chat_with_model(st.session_state['messages'])
    append_message("assistant", assistant_response)
    return assistant_response


st.title('Interactive Networking Image Analysis with GPT-4 Vision')

uploaded_file = st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'], key="file_uploader",
                                 help="Supported formats: PNG, JPG, JPEG")

if uploaded_file is not None:
    # Load the image using PIL
    image = Image.open(uploaded_file)
    # Extract width and height
    width, height = image.size
    st.write(f"Uploaded image dimensions: {width} x {height}")
    # Calculate tokens required for this image
    tokens_needed = calculate_image_tokens(width, height)
    st.write(f"Estimated token count for image processing: {tokens_needed}")

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

if uploaded_file is not None and 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = uploaded_file  # Save uploaded file into session state
    response = asyncio.run(handle_image_upload(st.session_state['uploaded_file']))
    with st.chat_message("assistant"):
        st.markdown(response)

prompt = st.chat_input("Say something", key="chat_input")
if prompt:
    append_message("user", prompt)
    response = asyncio.run(chat_with_model(st.session_state['messages']))
    append_message("assistant", response)
    display_messages()
