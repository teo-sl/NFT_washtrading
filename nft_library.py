"""
WARNING : functions that start with _ are private functions, they are not meant to be used outside this file.

PUBLIC FUNCTIONS:
    - topological_analysis              : analyze the topological graph structure, return a dictionary with the results
    - plot_degree_distribution          : plot the degree distribution of a graph
    - preprocess_data                   : initial preprocessing of the dataset
    - preprocess_repurchase             : (optional) to be called after preprocess_data, get only repurchase transactions
    - create_graph                      : create a graph from a dataframe. Need preprocess_data to be called first (optional preprocess_repurchase)
    - get_total_volume                  : get the total volume of transactions given a dataframe
    - get_wash_scc_mean_transaction     : detect wash trading transactions in the scc. Use the mean of the transactions for thresholding
    - get_wash_scc_volume_perc          : detect wash trading transactions in the scc. Use the volume percentage of the transactions for thresholding
    - get_dyadic_triadic_census         : get the dyadic and triadic census of a graph
    - analyze_dyads                     : return a dictionary with the result of the wash trading analysis on dyads
    - analyze_triads                    : return a dictionary with the result of the wash trading analysis on triads
    - get_reversing_pagerank_graph      : return the reversing pagerank graph
    - get_backbone_on_count             : return the backbone graph using count as strength

"""

### IMPORTS AND GLOBALS ###

import networkx as nx
import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
from backbone_network import get_graph_backbone

TRIAD_NAMES = [
                '003',
                '012',
                '102',
                '021D',
                '021U',
                '021C',
                '111D',
                '111U',
                '030T',
                '030C',
                '201',
                '120D',
                '120U',
                '120C',
                '210',
                '300',
]

### TOPOLOGICAL ANALYSIS ###

def topological_analysis(g : ig.Graph, path_measures=False):
    """
    This functions analyze the topological graph structure:
    it takes as input a graph (build with igraph) and a boolean variable.
    If the boolean variable is true, it also computes the diameter and the average path length.
    These two measures could be computationally expensive (they could be time-consuming),
    so if you don't need them, set the variable to false.

    """

    num_nodes = g.vcount()
    num_edges = g.ecount()
    density = g.density()
    avg_in_degree = sum(g.indegree())/num_nodes
    degree_assortativity = g.assortativity_degree()
    transitivity = g.transitivity_undirected()
    clustering_coefficient_avg= g.transitivity_avglocal_undirected(mode='zero')


    # get sink and source nodes
    sink_nodes = set(g.vs.select(_outdegree_eq=0))
    source_nodes = set(g.vs.select(_indegree_eq=0))

    # get the intersection between the two sets
    isolated_nodes = sink_nodes.intersection(source_nodes)
    percent_isolated_nodes = len(isolated_nodes)/num_nodes

    # get sink nodes \ isolated nodes
    sink_nodes = sink_nodes.difference(isolated_nodes)
    percent_sink_nodes = len(sink_nodes)/num_nodes

    # get source nodes \ isolated nodes
    source_nodes = source_nodes.difference(isolated_nodes)
    percent_source_nodes = len(source_nodes)/num_nodes

    #strongly connected component
    scc = g.clusters(mode='STRONG')
    num_strongly_components = scc.__len__()
    list_scc = scc.sizes()

    #weakly connected component
    wcc = g.clusters(mode='WEAK')
    num_weakly_components = wcc.__len__()
    list_wcc = wcc.sizes()

    diameter = None
    avg_path_length = None
    if path_measures:
        diameter = g.diameter()
        avg_path_length = g.average_path_length()

    print("Number of nodes: ", num_nodes)
    print("Number of edges: ", num_edges)
    print("Density: ", density)
    print("Average in degree: ", avg_in_degree)
    print("Degree assortativity: ", degree_assortativity)
    print("Transitivity: ", transitivity)
    print("Clustering coefficient (avg): ", clustering_coefficient_avg)
    print("Percent of sink nodes: ", percent_sink_nodes)
    print("Percent of source nodes: ", percent_source_nodes)
    print("Percent of isolated nodes: ", percent_isolated_nodes)
    print("Number of strongly connected components: ", num_strongly_components)
    print("Number of weakly connected components: ", num_weakly_components)
    if path_measures:
        print("Diameter: ", diameter)
        print("Average path length: ", avg_path_length)

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "avg_in_degree": avg_in_degree,
        "degree_assortativity": degree_assortativity,
        "transitivity": transitivity,
        "clustering_coefficient_avg": clustering_coefficient_avg,
        "percent_sink_nodes": percent_sink_nodes,
        "percent_source_nodes": percent_source_nodes,
        "percent_isolated_nodes": percent_isolated_nodes,
        "num_strongly_components": num_strongly_components,
        "num_weakly_components": num_weakly_components,
        "len_scc": list_scc,
        "len_wcc": list_wcc,
        "diameter": diameter,
        "avg_path_length": avg_path_length
    }

def plot_degree_distribution(G : nx.DiGraph):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees,log=True)
    plt.show()
    

### PREPROCESSING AND GRAPH CREATION ###

def preprocess_data(df : pd.DataFrame):
    """
    Preprocess the data, drop rows with missing values for seller and buyer,
    select only non-fungible tokens, convert timestamp to datetime, convert usd_price,
    create weight column as usd_price for readability, drop transaction associated with the null address
    """

    # drop rows with missing values for seller and buyer
    df = df.dropna(subset=['seller'], how='all')
    df = df.dropna(subset=['buyer'], how='all')
    # select only non-fungible tokens
    df = df[df['asset_contract_type']=='non-fungible']
    # convert timestamp to datetime
    df['tx_timestamp'] = pd.to_datetime(df['tx_timestamp'])
    # convert usd_price
    df['usd_price'] = df['usd_price'].astype(float)
    # create weight column as usd_price for readability
    df['weight'] = df['usd_price']
    # drop transaction associated with the null address
    df = df[df['buyer']!='0x0000000000000000000000000000000000000000'] 
    return df


def preprocess_repurchase(df : pd.DataFrame):
    """
    Preprocess the dataset to detect repurchase.
    This step is optional and need to be done after the preprocessing of the dataset,
    before the prepare_data_graph function.
    """

    # by ordering we can track if the buyer repurchase the same item in a  time window of 30 days
    df = df.sort_values(by=['buyer', 'opensea_url', 'tx_timestamp'])

    # use shift and diff to get the previous transaction (period=1)
    df['repurchase_sx'] = (df['buyer'].shift() == df['buyer']) & \
                   (df['opensea_url'].shift() == df['opensea_url']) & \
                   (df['tx_timestamp'].diff() < pd.Timedelta(days=30))
    df['repurchase_sx'] = df['repurchase_sx'].fillna(False)

    # use shift and diff to get the next transaction (period=-1)
    df['repurchase_dx'] = (df['buyer'].shift(-1) == df['buyer']) & \
                   (df['opensea_url'].shift(-1) == df['opensea_url']) & \
                   (df['tx_timestamp'].diff(-1) < pd.Timedelta(days=30))
    df['repurchase_dx'] = df['repurchase_dx'].fillna(False)

    # manage the edge case of the first and last transaction
    df['repurchase'] = df['repurchase_sx'] | df['repurchase_dx']
    # drop the temporary columns
    df = df.drop(['repurchase_sx', 'repurchase_dx'], axis=1)
    # get only the repurchase
    df = df[df['repurchase'] == True]
    return df


def _prepare_data_graph(df : pd.DataFrame, filter=None):
    """
    Prepare data for graph, select only seller, buyer, and weight columns,
    rename columns to from, to, and weight, group by from and to, and aggregate
    count, mean, and sum for weight. Optionally, apply filter to weight, select 
    only those transaction with weight>=filter.

    """
    df_graph = df[['seller', 'buyer', 'weight']]
    df_graph.columns = ['from', 'to', 'weight']
    df_graph = df_graph.groupby(['from', 'to']).agg({'weight': ['count', 'mean','sum']}).reset_index()
    df_graph.columns = ['from', 'to', 'count', 'mean','weight']
    # drop rows with negative weight
    df_graph = df_graph[df_graph['weight']>=0]
    if filter is not None:
        df_graph = df_graph[df_graph['weight']>=filter]
    return df_graph


def create_graph(df_preprocessed : pd.DataFrame ,use_networkx=True,filter=None):
    """
    Create the graph from the preprocessed data, optionally apply filter to weight.
    The dataset need to be preprocessed first using preprocess_data function (and optionally preprocess_repurchase function)
    You can also choose to create the graph using networkx or igraph.
    """
    df_graph = _prepare_data_graph(df_preprocessed,filter)
    if use_networkx:
        G = nx.from_pandas_edgelist(df_graph, 'from', 'to', ['count','mean','weight'], create_using=nx.DiGraph)
    else:
        G = ig.Graph.TupleList(df_graph.itertuples(index=False), directed=True, weights=False, edge_attrs=['count', 'mean', 'weight'])

    return G,df_graph


### SCC ANALYSIS ###

def get_total_volume(df : pd.DataFrame):
    """
    Get the total volume of the dataset by summing the weight column
    """
    return df['weight'].sum()

def _get_scc(graph_nx : nx.DiGraph):
    """
    Get the strongly connected components of the graph ordered by length,
    from the largest to the smallest.
    The graph need to be a networkx graph.
    """
    scc = [c for c in sorted(nx.strongly_connected_components(graph_nx), key=len, reverse=True)]
    return scc

def get_wash_scc_mean_transaction(graph_nx : nx.DiGraph, alpha=1.2):
    """
    Analyze the wash trading in the strongly connected components of the graph.
    The graph need to be a networkx graph.
    The alpha parameter is used to determine the threshold for the balance drift.
    This method detect wash trading by comparing the total balance drift
    from each node in the strongly connected components with the mean transaction. 
    """
    scc = _get_scc(graph_nx)
    volume_wash = 0
    suspected = []
    drifts = []
    for cc in scc:
        sg = graph_nx.subgraph(cc)
        
        # filter out component with no edges
        if sg.number_of_edges() == 0:
            continue
        
        total_volume = sum([e[2]['weight'] for e in sg.edges(data=True)])        
        if total_volume == 0:
            continue
        total_transactions = sum([e[2]['count'] for e in sg.edges(data=True)])

        mean_transaction = total_volume/total_transactions

        balances = []
        for node in sg.nodes:
            in_edges = sg.in_edges(node, data=True)
            out_edges = sg.out_edges(node, data=True)
            in_edges = [e for e in in_edges if e[0] in sg.nodes]
            out_edges = [e for e in out_edges if e[1] in sg.nodes]
            in_value = sum([e[2]['weight'] for e in in_edges])
            out_value = sum([e[2]['weight'] for e in out_edges])
            balances.append(abs((in_value-out_value)))
            
        balance_drift = sum(balances)/2.
        
        if balance_drift<mean_transaction*alpha:
            volume_wash += total_volume
            suspected.append(sg.nodes)
            drifts.append((balance_drift,mean_transaction))

    return {
        'volume_wash': volume_wash,
        'suspected': suspected,
        'num_suspected': len(suspected),
        'len_distribution' : [len(s) for s in suspected],
        'drift_mean_transaction' : drifts
    }


def get_wash_scc_volume_perc(graph_nx : nx.DiGraph, alpha=0.1):
    """
    Analyze the wash trading in the strongly connected components of the graph.
    The graph need to be a networkx graph.
    The alpha parameter is used to determine the threshold for the balance drift.
    This method detect wash trading by comparing the total balance drift
    from each node in the strongly connected components with the total volume.
    """

    scc = _get_scc(graph_nx)
    volume_wash = 0
    suspected = []
    drifts = []
    for cc in scc:
        sg = graph_nx.subgraph(cc)
        
        # filter out components with no edges
        if sg.number_of_edges() == 0:
            continue
        
        total_volume = sum([e[2]['weight'] for e in sg.edges(data=True)])
        
        if total_volume == 0:
            continue

        balances = []
        for node in sg.nodes:
            in_edges = sg.in_edges(node, data=True)
            out_edges = sg.out_edges(node, data=True)
            in_edges = [e for e in in_edges if e[0] in sg.nodes]
            out_edges = [e for e in out_edges if e[1] in sg.nodes]
            in_value = sum([e[2]['weight'] for e in in_edges])
            out_value = sum([e[2]['weight'] for e in out_edges])
            balances.append(abs((in_value-out_value)))
            
        balance_drift = sum(balances)/2.
        # normalize in [0,1]
        drift_perc = balance_drift/(total_volume)
    
        if drift_perc<alpha:
            volume_wash += total_volume
            suspected.append(sg.nodes)
            

    return {
        'volume_wash': volume_wash,
        'suspected': suspected,
        'num_suspected': len(suspected),
        'len_distribution' : [len(s) for s in suspected],
        'drift_perc' : drifts
    }

### MOTIFS ANALYSIS ###

def _convert_list_to_triad_dict(triad_list : list):
    """
    Convert the igraph triad census in a more readable way
    """
    triad_dict = {}
    for i,triad in enumerate(triad_list):
        triad_dict[TRIAD_NAMES[i]] = triad_list[i]
    return triad_dict

def get_dyadic_tryadic_census(g : ig.Graph):
    """
    Get the dyadic and triadic census of the graph using igraph.
    """
    triads = g.triad_census()
    dyads = g.dyad_census()
    dyads_dict = {'mutual':dyads[0],'asymmetric':dyads[1],'null':dyads[2]}
    triads_dict = _convert_list_to_triad_dict(triads)
    return {
        'dyads': dyads_dict,
        'triads': triads_dict
    }

#### DYADS ANALYSIS ####

def _get_mutual_dyads(g : ig.Graph):
    """
    Get the list of mutual dyads in the graph.
    """
    reciprocal_nodes = []
    for edge in g.es:
        to_add = set([edge.target, edge.source])
        if  g.are_connected(edge.target, edge.source)  and not to_add in reciprocal_nodes:
            reciprocal_nodes.append(to_add)
    return reciprocal_nodes


def analyze_dyads(g : ig.Graph, use_mean_transaction=True, alpha=1.2):
    """
    Analyze the dyads of the graph.
    The alpha parameter is used to determine the threshold for the balance drift.
    This method detect wash trading by comparing the total balance drift or 
    the mean transaction value, you can choose which one to use by setting the
    use_mean_transaction parameter.
    """
    reciprocal_nodes = _get_mutual_dyads(g)
    suspected_volume = 0
    suspected_dyads = [] 
    for nodes in reciprocal_nodes:
        n1=nodes.pop()
        
        # manage self loops
        if len(nodes) != 1: n2=n1
        else: n2 = nodes.pop()

        e1 = g.get_eid(n1, n2)
        e2 = g.get_eid(n2, n1)

        w1 = g.es[e1]['weight']
        w2 = g.es[e2]['weight']

        total_volume = w1+w2
        if use_mean_transaction:
            c1 = g.es[e1]['count']
            c2 = g.es[e2]['count']
            single_transaction_value = (w1+w2)/(c1+c2)
        
        # filter out zero transactions
        if total_volume==0:
            continue

        drift_perc = abs(w1-w2)/(2*total_volume)

        if ((use_mean_transaction and abs(w1-w2)<single_transaction_value*alpha) or 
                                (not use_mean_transaction and drift_perc<alpha)):
            suspected_volume+=total_volume
            suspected_dyads.append(set([n1,n2]))
    return {
        'suspected' : suspected_dyads,
        'volume_wash' : suspected_volume,
        'num_suspected' : len(suspected_dyads)
    }

#### TRIADS ANALYSIS ####

def _find_wash_triangles(g,triads,alpha=1.2):
    """
    Find wash trading triangles in the graph. This function use 
    the mean transaction value to determine the threshold for the balance drift.
    In addition alpha is used to make the threshold more strict or more loose.
    """
    suspected = []
    for triad in triads:
        sg = g.subgraph(triad)
        balances = [0 for _ in range(3)]
        w_tot = sum([e['weight'] for e in sg.es])
        if w_tot == 0:
            continue
        c_tot = sum ([e['count'] for e in sg.es])
        mean_weight_per_transaction = w_tot/c_tot
        for edge in sg.es:
            n1 = edge.source
            n2 = edge.target
            w = edge['weight']
            balances[n1] += w
            balances[n2] -= w
        balances_abs = [abs(b) for b in balances]   
        sum_balances = sum(balances_abs)/2
        if sum_balances <= mean_weight_per_transaction*alpha:
            suspected.append(triad)
    return {
        'suspected' : suspected,
        'volume_wash' : w_tot,
        'num_suspected' : len(suspected)   

    }


def _find_wash_common(g,triads,is_founder=True,alpha=1.2):
    """
    Find wash trading in the graph checking the common founder or common exit phenomenon.
    This function use the mean transaction value to determine the threshold for the balance drift.
    In addition alpha is used to make the threshold more strict or more loose.
    """
    suspected_common = []
    volume_wash = 0
    for cf in triads:
        # find the nodes that are reciprocally connected
        n1 = cf[0]
        n2 = cf[1]
        n3 = cf[2]
        reciprocal_nodes = []
        if g.are_connected(n1, n2) and g.are_connected(n2, n1):
            reciprocal_nodes = [n1,n2]
        elif g.are_connected(n1, n3) and g.are_connected(n3, n1):
            reciprocal_nodes = [n1,n3]
        elif g.are_connected(n2, n3) and g.are_connected(n3, n2):
            reciprocal_nodes = [n2,n3]
        else:
            continue
        rn1 = reciprocal_nodes[0]
        rn2 = reciprocal_nodes[1]
        # get the other node
        other_node = [n for n in cf if n not in reciprocal_nodes][0]
        if is_founder:
            e1 = g.get_eid(other_node, rn1)
            e2 = g.get_eid(other_node, rn2)
        else:
            e1 = g.get_eid(rn1, other_node)
            e2 = g.get_eid(rn2, other_node)
        w1 = g.es[e1]['weight']
        w2 = g.es[e2]['weight']
        c1 = g.es[e1]['count']
        c2 = g.es[e2]['count']

        single_transaction_value = (w1+w2)/(c1+c2)
        total_volume = w1+w2

        if total_volume==0:
            continue

        if abs(w1-w2)<single_transaction_value*alpha:
            suspected_common.append(cf)
            volume_wash+=total_volume

    return {
        'suspected' : suspected_common,
        'volume_wash' : volume_wash,
        'num_suspected' : len(suspected_common)   
    }

def _check_type(g : ig.Graph, triad):
    """
    Return the type of the triad as one of the names in the TRIAD_NAMES list.
    """
    sg = g.subgraph(triad)
    triad_type = sg.triad_census()
    idx = [i for i, x in enumerate(triad_type) if x != 0]
    assert len(idx) == 1
    return TRIAD_NAMES[idx[0]]

def analyze_triads(g : ig.Graph, alpha=1.2):
    """
    Analyze the wash trading phenomenon in the graph g by analyzing the triads.
    The function focuses on particular type of triads, the first group of triads
    refer to cycles, the second revolves around the phenomenons of common founder and
    common exit.
    """
    ret = {}

    # manage common founder and common exit
    patterns = ["A<-B->C, A<->C","A->B<-C, A<->C"]
    names = ["120D","120U"]
    results = {}
    for i,p in enumerate(patterns):
        print(f'Starting {names[i]}')
        ffl = ig.Graph.Formula(p)
        triads = g.get_subisomorphisms_lad(ffl, induced=True)
        # remove duplicates
        triads = [set(r) for r in triads]
        triads = list(set(map(frozenset, triads)))
        triads = [list(t) for t in triads]
        is_founder = names[i] == "120D" 
        result = _find_wash_common(g,triads,is_founder,alpha)
        results[names[i]] = result
        print(f'Finished {names[i]}')
        print('------------------')
    ret['common'] = results

    # manage cyclic triads
    patterns = ["A<->B<->C","A<-B<-C, A->C","A->B->C, A<->C","A->B<->C, A<->C","A<->B<->C, A<->C"]
    names = ["201","030C","120C","210","300"]
    results = {}
    for i,p in enumerate(patterns):
        print(f'Starting {names[i]}')
        ffl = ig.Graph.Formula(p)
        triad = g.get_subisomorphisms_lad(ffl, induced=True)
        # remove duplicates
        triad = [set(r) for r in triad]
        triad = list(set(map(frozenset, triad)))
        result = _find_wash_triangles(g,triad,alpha)
        results[names[i]] = result
        print(f'Finished {names[i]}')
        print('------------------')
    ret['cycles'] = results

    return ret
    

### REVERSE - COUNT BASED BACKBONE ###

def get_reversing_pagerank_graph(G : nx.DiGraph,percentile=0.1):
    """
    This function compute the subgraph from the 10th percentile of the pagerank
    applied to the reverse graph of G. The percentile can be changed by setting
    the percentile parameter. Once the graph is computed, it is reversed again.
    From this graph the user can launch the backbone algorithm and then make
    wash trading analysis.
    """
    G_inverse = G.reverse()
    pr = nx.pagerank(G_inverse,weight='weight')
    pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    pr_selected = pr_sorted[:int(len(pr_sorted)*percentile)]
    G_sg = G.subgraph([x[0] for x in pr_selected])
    G_sg = G_sg.reverse()
    return G_sg


def get_backbone_on_count(G : nx.DiGraph,alpha_t=0.05):
    """
    This function compute the backbone of the graph G based on the count. 
    First, change the weight of the edges to the count and then apply the
    backbone algorithm. Once the backbone is computed, the weight is changed
    back to the volume.
    """
    # convert weight to volume and count to weight
    for u, v, data in G.edges(data=True):
        data["volume"] = data.pop("weight")
        data["weight"] = data.pop("count")
    G_bb_count = get_graph_backbone(G,alpha_t=alpha_t)
    # reconvert back edges attributes
    for u, v, data in G_bb_count.edges(data=True):
        data["count"] = data.pop("weight")
        data["weight"] = data.pop("volume")
    return G_bb_count
