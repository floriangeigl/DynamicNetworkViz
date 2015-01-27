__author__ = 'Florian'
import datetime
import pandas as pd
from graph_tool.all import *
from dyn_net_viz import graph_viz
import random


def main():
    # create an undirected tree
    second_level_nodes = 3
    second_level_leaves = 1
    g = Graph(directed=False)
    vertices = [g.add_vertex() for i in range(second_level_nodes + 1)]
    root_vertex, second_level = vertices[0], vertices[1:]
    third_level = []
    for idx, v2 in enumerate(second_level):
        g.add_edge(root_vertex, v2)
        for ov2 in second_level[idx + 1:]:
            g.add_edge(v2, ov2)
        # add leaves to each second level vertex
        for i in range(second_level_leaves):
            v = g.add_vertex()
            g.add_edge(v2, v)
            if i > 0:
                g.add_edge(v, third_level[-1])
            third_level.append(v)
        g.add_edge(v2, g.add_vertex())
    print g

    data = []
    iteration = 0
    # init root vertex as active
    data.append((iteration, root_vertex, 300))

    # in each iteration add one of the second level nodes
    iteration += 1
    for v in second_level:
        data.append((iteration, v, 200))

    # in each iteration add one of the third level nodes
    iteration += 1
    for v in third_level:
        data.append((iteration, v, 100))

    for i in range(50):
        iteration += 1
        v = random.choice(third_level)
        newv = g.add_vertex()
        g.add_edge(v, newv)
        data.append((iteration, newv, 100))

    # create the dataframe
    df = pd.DataFrame(columns=['iteration', 'vertex', 'visible'], data=data)
    plot_each = 1
    # create graph-viz-obj
    gv = graph_viz(df, g, filename='example_evo_output/graph_evolution', cmap='jet', df_iteration_key='iteration',
                   df_vertex_key='vertex', df_state_key='visible', plot_each=plot_each, smoothing=5, max_node_alpha=1.0,
                   output_size=(800, 600))

    # activate the dynamic positioning flag
    gv.plot_network_evolution(dynamic_pos=True)


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print '============================='
    print 'All done. Overall Time:', str(datetime.datetime.now() - start)