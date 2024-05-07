from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.runnables.base import RunnableSerializable


def build_reflection_chain(chat_model: BaseChatModel) -> RunnableSerializable:
    prompt_critic_system_message = (
        "You are an expert critic specializing in the analysis of network topology diagrams, "
        "renowned for your meticulous attention to detail. Your task involves reviewing a "
        "given network diagram and its accompanying analysis. You are expected to identify "
        "inaccuracies and suggest detailed improvements. Ensure that your critique covers "
        "the accuracy of the data flow representation from end clients, the routing protocols "
        "used, the networking devices involved, and layer 2 details such as VLAN "
        "configurations. Your goal is to ensure the analysis is both thorough and precise."
    )
    reflection_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt_critic_system_message,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    reflect = reflection_prompt | chat_model
    return reflect


def build_writer_chain(chat_model: BaseChatModel) -> RunnableSerializable:
    writer_system_message = (
        "You are an expert in analyzing network topology diagrams. Your task is to provide a detailed "
        "explanation of the given topology diagram, highlighting the specific products used, and, "
        "where applicable, include an analysis of VLAN configurations, OSPF routing areas, BGP ASNs, "
        "and other relevant network features. Additionally, you will describe the data flow within "
        "the network, starting from the end clients and detailing the interactions between various "
        "network elements. "

        "Please ensure that any feedback is directly integrated into the existing analysis, refining and "
        "enhancing the information rather than merely commenting on the feedback.")
    writer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                writer_system_message,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    writer = writer_prompt | chat_model
    return writer
