__author__ = 'Florian'
from dyn_net_viz import graph_viz
import datetime
import pandas as pd
from graph_tool.all import *


def main():
    # create an circular undirected graph containing 10 nodes
    num_nodes = 10
    g = Graph(directed=False)
    nodes = [g.add_vertex() for i in range(num_nodes)]
    for idx, v1 in enumerate(nodes):
        g.add_edge(v1, nodes[(idx + 1) % len(nodes)])
    print g

    # define two opinions and pass one of them from neighbour to neighbour through the graph
    op1 = 0
    op2 = 1
    data = []
    # first iteration, each node has opinion 1
    iteration = 0
    for v in nodes:
        data.append((iteration, v, {op1}))
    iteration += 1

    # pass op2 through the graph
    for idx, v in enumerate(nodes):
        iteration += 1
        data.append((iteration, v, {op1, op2}))
        data.append((iteration, nodes[idx - 1], {op1}))

    # remove all opinions
    for v in nodes:
        iteration += 1
        data.append((iteration, v, {}))

    #create the dataframe
    df = pd.DataFrame(columns=['iteration', 'vertex', 'opinion'], data=data)
    print df

    #create graph-viz-obj
    gv = graph_viz(df, g, filename='example_opdyn_output/opinion_dynamic', df_iteration_key='iteration',
                   df_vertex_key='vertex', df_opinion_key='opinion', ips=1, smoothing=10, max_node_alpha=1.0,
                   output_size=(800, 600), edge_blending=True)
    gv.plot_network_evolution()


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print '============================='
    print 'All done. Overall Time:', str(datetime.datetime.now() - start)