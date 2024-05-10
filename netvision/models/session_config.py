import os
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from typing import Dict, Optional, List, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables.base import RunnableSerializable
from dotenv import load_dotenv


class SessionConfig(BaseModel):
    chat_model: Optional[BaseChatModel] = None
    compiled_graph: Optional[CompiledGraph] = None
    writer_chain: Optional[RunnableSerializable] = None
    reflection_chain: Optional[RunnableSerializable] = None
    messages: List[BaseMessage] = []
    images: Dict[str, Any] = {}
    initialized: bool = False  # Flag to check if initialization has been done

    def __init__(self, /, *args, **kwargs):
        # Important to call super().__init__(**kwargs) to let Pydantic do its setup
        super().__init__(**kwargs)
        if not self.chat_model:
            self.chat_model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("OPENAI_MODEL_NAME"))

    def add_message(self, message: BaseMessage):
        """Append a message to the messages list."""
        self.messages.append(message)

    def add_messages(self, messages: List[BaseMessage]):
        """Extend the messages list with multiple messages."""
        self.messages.extend(messages)


load_dotenv(override=True)
session_config = SessionConfig()
