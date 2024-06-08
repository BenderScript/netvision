import os
from pydantic import BaseModel
from typing import Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables.base import RunnableSerializable
from dotenv import load_dotenv


class Config(BaseModel):
    initialized: bool = False  # Flag to check if initialization has been done


# load_dotenv(override=True)
config = Config()
