import asyncio
import os
import pprint
from io import BytesIO

import openai
from openai import AsyncOpenAI
import streamlit as st
import base64
from dotenv import load_dotenv
from PIL import Image

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages.base import messages_to_dict
from netvision.chains.builder import build_reflection_chain, build_writer_chain
from netvision.graph.builder import build_graph
from netvision.image.process import calculate_image_tokens, encode_image_with_mime_type
from netvision.models.config import Config

load_dotenv(override=True)

# Set your OpenAI API key here
load_dotenv(override=True)
st.session_state['openai_client'] = AsyncOpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = st.session_state['openai_client']
# os.environ["LANGCHAIN_PROJECT"] = "Netvision-Reflection"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"


if not st.session_state.get('ChatOpenAI'):
    st.session_state['ChatOpenAI'] = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4-turbo")

if not st.session_state.get('netvision_config'):
    netvision_config = Config()
    netvision_config.reflection_chain = build_reflection_chain(netvision_config.chat_model)
    netvision_config.writer_chain = build_writer_chain(netvision_config.chat_model)
    netvision_config.compiled_graph = build_graph()
    st.session_state['netvision_config'] = netvision_config


def get_image_dimensions(image_file):
    # Load the image using PIL
    image = Image.open(uploaded_file)
    # Extract width and height
    width, height = image.size
    st.write(f"Uploaded image dimensions: {width} x {height}")
    return width, height


# https://community.openai.com/t/how-do-i-calculate-image-tokens-in-gpt4-vision/492318


# Specific to streamlit, image_file in this case would be an uploaded file object


async def process_events(human_prompt, config):
    events = []
    async for event in netvision_config.compiled_graph.astream(
            input={"input": human_prompt, "messages": human_prompt}, config=config
    ):
        events.append(event)
        if "generate" in event:
            pprint.pprint(event.get("generate").get("messages"))
        elif "reflect" in event:
            pprint.pprint(event.get("reflect").get("messages"))
        else:
            pprint.pprint(event)
        print("---")
    return events


# https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format
# finish_reason = "length"
def display_messages():
    if 'messages' in st.session_state:
        for msg in st.session_state['messages']:
            with st.chat_message(msg.type):
                if isinstance(msg.content, list):  # Check if content is a list of items
                    for content in msg.content:
                        if content["type"] == 'text':
                            st.markdown(content['text'])
                        elif content["type"] == 'image_url':
                            # Assuming the image data comes directly in a base64 format
                            base64_image = content['image_url']['url']
                            header, encoded = base64_image.split(",", 1)

                            # Decode the image data
                            data = base64.b64decode(encoded)

                            # Create a PIL Image instance
                            image = Image.open(BytesIO(data))
                            st.image(image, caption='Uploaded Image', use_column_width=True)
                else:  # Handle the case where content is a simple string
                    st.markdown(msg.content)


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

    human_prompt = [HumanMessage(content=[
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

    append_message_list(human_prompt)
    return human_prompt


# https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format
# finish_reason="length"
async def chat_with_model(session_messages):
    try:
        with st.spinner('Working on your prompt, please wait...'):
            chat_model = st.session_state['ChatOpenAI']
            response = await chat_model.ainvoke(session_messages)
            content = response.content
            finish_reason = response.response_metadata.get("finish_reason")

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


st.title('Interactive Networking Image Analysis with GPT-4 Vision')

with st.sidebar:
    if uploaded_file := st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'], key="file_uploader",
                                         help="Supported formats: PNG, JPG, JPEG"):

        with st.status("Working your image..."):
            st.write("Analyzing image...")
            w, h = get_image_dimensions(uploaded_file)
            tokens_needed = calculate_image_tokens(w, h)
            st.write(f"Estimated token count for image processing: {tokens_needed}")

            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            image_key = uploaded_file.name + str(uploaded_file.size)

            if image_key not in st.session_state:
                st.session_state[image_key] = uploaded_file  # Save uploaded file into session state
                st.write("Creating Agent Prompt...")
                initial_prompt = asyncio.run(handle_image_upload_agents(st.session_state[image_key]))
                netvision_config = st.session_state['netvision_config']
                st.write("Communicating with Agents...")
                events = asyncio.run(process_events(human_prompt=initial_prompt,
                                                    config={"configurable": netvision_config.dict()}))
                append_message_list(events[-1].get("generate").get("messages"))
                st.write("Analysis complete!")

display_messages()

if user_prompt := st.chat_input("Say something", key="chat_input"):
    append_message_list([HumanMessage(content=[user_prompt])])
    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.chat_message("assistant"):
        response = asyncio.run(chat_with_model(st.session_state['messages']))
        st.markdown(response)
        append_message_list([AIMessage(response)])
