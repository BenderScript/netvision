import os
from pydantic import BaseModel
from typing import Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables.base import RunnableSerializable


class Config(BaseModel):
    chat_model: Optional[BaseChatModel] = None
    compiled_graph: Optional[CompiledGraph] = None
    writer_chain: Optional[RunnableSerializable] = None
    reflection_chain: Optional[RunnableSerializable] = None

    def __init__(self, /, *args, **kwargs):
        super().__init__(**kwargs)
        if not self.chat_model:
            self.chat_model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("OPENAI_MODEL_NAME"))

