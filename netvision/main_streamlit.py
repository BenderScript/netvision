import asyncio
import mimetypes
import os
import pprint
from io import BytesIO
from math import ceil

import streamlit as st
import base64
from dotenv import load_dotenv
from PIL import Image

from langchain_core.messages import HumanMessage

from netvision.chains.builder import build_reflection_chain, build_writer_chain
from netvision.graph.builder import build_graph
from netvision.models.config import Config

load_dotenv(override=True)


# os.environ["LANGCHAIN_PROJECT"] = "Netvision-Reflection"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"


# if not st.session_state.get('ChatOpenAI'):
#     st.session_state['ChatOpenAI'] = ChatOpenAI(api_key= os.getenv("OPENAI_API_KEY"), model="gpt-4-turbo")


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


# Specific to streamlit, image_file in this case would be an uploaded file object
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


netvision_config = Config()
netvision_config.reflection_chain = build_reflection_chain(netvision_config.chat_model)
netvision_config.writer_chain = build_writer_chain(netvision_config.chat_model)
netvision_config.compiled_graph = build_graph()
config = {"configurable": netvision_config.dict()}


async def process_events():
    async for event in netvision_config.compiled_graph.astream(
            input={"input": prompt, "messages": prompt}, config=config
    ):
        if "generate" in event:
            pprint.pprint(event.get("generate").get("messages"))
        elif "reflect" in event:
            pprint.pprint(event.get("reflect").get("messages"))
        else:
            pprint.pprint(event)
        print("---")


asyncio.run(process_events())


# https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format
finish_reason = "length"


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


async def handle_image_upload_agents(file):
    """Process and display the uploaded image and provides analysis using agents."""
    data_uri = encode_image_with_mime_type(file)
    st.session_state['image_data'] = data_uri  # Store the encoded image in session state

    prompt = [HumanMessage(content=[
        {
            "type": "text",
            "text": "Analyze this network diagram"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": data_uri
            }
        }
    ])]

    append_message_list(prompt)
    return prompt

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
            response = asyncio.run(handle_image_upload_agents(st.session_state[image_key]))

display_messages()

if prompt := st.chat_input("Say something", key="chat_input"):
    append_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = asyncio.run(chat_with_model(st.session_state['messages']))
        st.markdown(response)
        append_message("assistant", response)


