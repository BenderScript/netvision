import asyncio
import pprint
from io import BytesIO

import openai
import streamlit as st
import base64
from dotenv import load_dotenv
from PIL import Image

from langchain_core.messages import HumanMessage, AIMessage
from netvision.image.process import calculate_image_tokens, encode_image_with_mime_type, get_image_dimensions
from netvision.models.session_config import session_config
from netvision.services.initializer import initialize_components

load_dotenv(override=True)

# os.environ["LANGCHAIN_PROJECT"] = "Netvision-Reflection"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"


async def process_events(human_prompt, agent_config):
    events = []
    async for event in netvision_session_config.compiled_graph.astream(
            input={"input": human_prompt, "messages": human_prompt}, config=agent_config
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
    for msg in netvision_session_config.messages:
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


async def handle_image_upload_agents(file):
    """Process and display the uploaded image and provides analysis using agents."""
    data_uri = encode_image_with_mime_type(file)

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

    return human_prompt


# https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format
# finish_reason="length"
async def chat_with_model(session_messages):
    try:
        with st.spinner('Working on your prompt, please wait...'):
            chat_model = netvision_session_config.chat_model
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

netvision_session_config = initialize_components(session_config)
st.title('NetVision - Networking Diagram Analysis')

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

            if image_key not in netvision_session_config.images:
                netvision_session_config.images[image_key] = uploaded_file  # Save uploaded file into session state
                st.write("Creating Agent Prompt...")
                initial_prompt = asyncio.run(handle_image_upload_agents(uploaded_file))
                netvision_session_config.add_messages(initial_prompt)
                st.write("Communicating with Agents...")
                events = asyncio.run(process_events(human_prompt=initial_prompt,
                                                    agent_config={"configurable": netvision_session_config.dict()}))
                netvision_session_config.add_messages(events[-1].get("generate").get("messages"))
                st.write("Analysis complete!")

display_messages()

if user_prompt := st.chat_input("Say something", key="chat_input"):
    netvision_session_config.add_messages([HumanMessage(content=[user_prompt])])
    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.chat_message("assistant"):
        response = asyncio.run(chat_with_model(netvision_session_config.messages))
        st.markdown(response)
        netvision_session_config.add_messages([AIMessage(response)])
