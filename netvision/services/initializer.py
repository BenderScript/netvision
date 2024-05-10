from netvision.chains.builder import build_reflection_chain, build_writer_chain
from netvision.graph.builder import build_graph


def initialize_components(session_config):
    if session_config.initialized:
        return session_config
    session_config.reflection_chain = build_reflection_chain(session_config.chat_model)
    session_config.writer_chain = build_writer_chain(session_config.chat_model)
    session_config.compiled_graph = build_graph()
    session_config.initialized = True
    return session_config
