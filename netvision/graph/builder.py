from langgraph.graph import StateGraph, END
from netvision.graph.nodes import generation_node, reflection_node, AgentState


def should_continue(state: AgentState):
    if len(state["messages"]) > 3:
        # End after 3 iterations
        return END
    return "reflect"


def build_graph():

    # builder = MessageGraph()
    builder = StateGraph(AgentState)
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.set_entry_point("generate")
    builder.add_conditional_edges("generate", should_continue)
    builder.add_edge("reflect", "generate")
    # Compiled graph has access to nodes and edges
    return builder.compile()
