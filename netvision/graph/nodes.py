import operator
from typing import Sequence, List, Any, Dict, TypedDict, Union, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.agents import AgentAction, AgentFinish


class AgentState(TypedDict):
    # The input string
    input: str
    # The list of previous messages in the conversation
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    last_agent: str


async def generation_node(state: AgentState, config: Dict) -> Dict[str, list]:
    # Other messages we need to adjust
    writer_chain = config['configurable'].get('writer_chain')
    messages = state['messages']
    response = await writer_chain.ainvoke(messages)
    return {"messages": [response]}


async def reflection_node(state: AgentState, config: Dict) ->  Dict[str, list]:
    # Other messages we need to adjust
    reflection_chain = config['configurable'].get('reflection_chain')
    messages = state['messages']
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [messages[0]] + [
        cls_map[msg.type](content=msg.content) for msg in messages[1:]
    ]
    # Not documented in Langgraph, but this is how we call the model
    response = await reflection_chain.ainvoke({"messages": translated})
    # We treat the output of this as human feedback for the generator
    return {"messages": [response]}
