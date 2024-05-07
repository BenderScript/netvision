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

# Set your OpenAI API key here
load_dotenv(override=True)
#os.environ["LANGCHAIN_PROJECT"] = "Netvision-Reflection"
#os.environ["LANGCHAIN_TRACING_V2"] = "true"


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


prompt = [HumanMessage(content=[
    {
        "type": "text",
        "text": "Analyze this network diagram"
    },
    {
        "type": "image_url",
        "image_url": {
            "url": encode_image_path_with_mime_type("../resources/imgs/diagram1.png")
        }
    }
])]

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
