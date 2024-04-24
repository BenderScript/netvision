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
    except openai.BadRequestError as e:  # Don't forget to add openai
        # Handle error 400
        print(f"Error 400: {e}")
    except openai.AuthenticationError as e:  # Don't forget to add openai
        # Handle error 401
        print(f"Error 401: {e}")
    except openai.PermissionDeniedError as e:  # Don't forget to add openai
        # Handle error 403
        print(f"Error 403: {e}")
    except openai.NotFoundError as e:  # Don't forget to add openai
        # Handle error 404
        print(f"Error 404: {e}")
    except openai.UnprocessableEntityError as e:  # Don't forget to add openai
        # Handle error 422
        print(f"Error 422: {e}")
    except openai.RateLimitError as e:  # Don't forget to add openai
        # Handle error 429
        print(f"Error 429: {e}")
    except openai.InternalServerError as e:  # Don't forget to add openai
        # Handle error >=500
        print(f"Error >=500: {e}")
    except openai.APIConnectionError as e:  # Don't forget to add openai
        # Handle API connection error
        print(f"API connection error: {e}")
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


with st.sidebar:
    if uploaded_file := st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'], key="file_uploader",
                                     help="Supported formats: PNG, JPG, JPEG"):
        # Load the image using PIL
        image = Image.open(uploaded_file)
        # Extract width and height
        width, height = image.size
        st.write(f"Uploaded image dimensions: {width} x {height}")
        # Calculate tokens required for this image
        tokens_needed = calculate_image_tokens(width, height)
        st.write(f"Estimated token count for image processing: {tokens_needed}")

        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        image_key = uploaded_file.name + str(uploaded_file.size)

        if image_key not in st.session_state:
            st.session_state[image_key] = uploaded_file  # Save uploaded file into session state
            response = asyncio.run(handle_image_upload(st.session_state[image_key]))


display_messages()

if prompt := st.chat_input("Say something", key="chat_input"):
    append_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = asyncio.run(chat_with_model(st.session_state['messages']))
        st.markdown(response)
        append_message("assistant", response)
