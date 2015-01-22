__author__ = 'Florian'
import datetime
import pandas as pd
from graph_tool.all import *
from dyn_net_viz import graph_viz

def main():
    # create an undirected tree
    second_level_nodes = 10
    second_level_leaves = 2
    g = Graph(directed=False)
    vertices = [g.add_vertex() for i in range(second_level_nodes + 1)]
    root_vertex, second_level = vertices[0], vertices[1:]
    third_level = []
    for v2 in second_level:
        g.add_edge(root_vertex, v2)
        # add to each second level vertex leaves
        for i in range(second_level_leaves):
            v = g.add_vertex()
            g.add_edge(v2, v)
            third_level.append(v)
        g.add_edge(v2, g.add_vertex())
    print g

    # init all nodes with activity = 0
    data = []
    iteration = 0
    for v in g.vertices():
        data.append((iteration, v, 0))

    for i in range(3):
        # set activity of root vertex to 100
        iteration += 1
        data.append((iteration, root_vertex, 100))

        # set activity of all second-level vertices to 100, reduce root vertex activity to 50
        iteration += 1
        for v in second_level:
            data.append((iteration, v, 100))
        data.append((iteration, root_vertex, 50))

        # set activity of all third-level vertices to 100, reduce second-level vertices activity to 50, set root vertex activity to 0
        iteration += 1
        for v in third_level:
            data.append((iteration, v, 100))
        for v in second_level:
            data.append((iteration, v, 50))
        data.append((iteration, root_vertex, 0))

        # and so forth and so on
        iteration += 1
        for v in third_level:
            data.append((iteration, v, 50))
        for v in second_level:
            data.append((iteration, v, 0))
        iteration += 1
        for v in third_level:
            data.append((iteration, v, 0))

    #create the dataframe
    df = pd.DataFrame(columns=['iteration', 'vertex', 'activity'], data=data)

    #create graph-viz-obj
    gv = graph_viz(df, g, filename='example2_output/activity_dynamics', df_iteration_key='iteration', df_vertex_key='vertex', df_state_key='activity', fps=1, smoothing=10, max_node_alpha=1.0, output_size=(800, 600), edge_blending=True)
    gv.plot_network_evolution()


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print '============================='
    print 'All done. Overall Time:', str(datetime.datetime.now() - start)