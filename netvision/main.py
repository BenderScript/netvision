import asyncio
import pprint
import openai

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage

from netvision.chains.builder import build_reflection_chain, build_writer_chain
from netvision.graph.builder import build_graph
from netvision.image.process import encode_image_with_mime_type, encode_image_path_with_mime_type
from netvision.models.config import Config

# Set your OpenAI API key here
load_dotenv(override=True)
# os.environ["LANGCHAIN_PROJECT"] = "Netvision-Reflection"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

netvision_config = Config()
netvision_config.reflection_chain = build_reflection_chain(netvision_config.chat_model)
netvision_config.writer_chain = build_writer_chain(netvision_config.chat_model)
netvision_config.compiled_graph = build_graph()


async def handle_image_upload_agents(file):
    """Process and display the uploaded image and provides analysis using agents."""
    data_uri = encode_image_path_with_mime_type(file)

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


async def chat_with_model(session_messages):
    try:
        chat_model = netvision_config.chat_model
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

initial_prompt = handle_image_upload_agents("../resources/imgs/diagram1.png")
asyncio.run(process_events(human_prompt=initial_prompt,
                           config={"configurable": netvision_config.dict()}))
