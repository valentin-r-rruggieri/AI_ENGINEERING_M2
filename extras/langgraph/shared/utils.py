# Este modulo construye grafos locales pequenos para aprender LangGraph sin API key ni modelo externo.
# Los scripts llaman esta funcion con un concepto y observan como cambia el estado de un grafo.

# StateGraph, START y END definen un flujo de nodos con estado compartido.
from langgraph.graph import END, START, StateGraph

def run_lesson(topic: str) -> None:
    def node(state: dict) -> dict:
        return {"topic": topic, "steps": state.get("steps", 0) + 1}
    graph = StateGraph(dict)
    graph.add_node("lesson", node)
    graph.add_edge(START, "lesson")
    graph.add_edge("lesson", END)
    print(graph.compile().invoke({"steps": 0}))

# Resumen final: LangGraph permite expresar agentes como nodos y transiciones sobre estado.
