import pygraphviz as pgv
from networkx.drawing.nx_agraph import to_agraph

from primatives import Graph

def draw_graph(g: Graph, path: str):
    A = to_agraph(g.nx)
    A.layout(prog="dot")
    A.draw(path)
