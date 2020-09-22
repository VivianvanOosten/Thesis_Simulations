import networkx as nx

#Graph is a collection of nodes and edges.
G = nx.Graph()

#In general: NetworkX stores stuff in a dictionary of dictionaries 
# where the outer is {node: value, node: value, etc}, and the 
# value is {neighbor: edge attributes, neighbor: edge attributes, etc}

#Can add nodes from a list
#including attributes with (node, attribute dictionary)
G.add_nodes_from([1,2, (3, {"color": "red"})  ] )

#Can add edges from a list too
G.add_edges_from( [ (1,2) , (1,3) ])

#Double nodes and edges are allowed

#Graph information is given by
G.nodes
G.edges
G.adj.items() #returns []
G.degree
G.edges.items() #returns a list of tuples [ (edge, attributes), (edge, attributes)]
G[u] #Gives neighbors of u
G.edges[u, v] #Gives edges between u and v
G.edges.data('attribute') #Gives a list of tuples [(edge, attribute), (edge,attribute)]

#Can remove elements from the graph
G.remove_nodes_from([1, 2])

#Creating a graph can be done from a list of edges
edgelist = [ (0,1), (1,2), (2,3) ]
H = nx.Graph(edgelist)

#Accessing parts of the graphs. 
G[1] #= list of neighbors of 1
G.adj.items() #= list of (node, neighbors) for every node

#Multigraphshttps://networkx.github.io/documentation/stable/reference/generators.html 
MG = nx.MultiGraph() #initialising an empty graph

#Generating graphs
#There are many options for random graphs
#I should just look up which one I want to use. 
#Other generating methods can be found in 

#Analyzing graphs
list(nx.connected_components(G)) #prints all connected components
#Other algorithms can be found in https://networkx.github.io/documentation/stable/reference/algorithms/index.html



#Specifically for the configuration model!
sequence = nx.random_powerlaw_tree_seqence(100, tries = '5000')
    #could choose any random sequence generator
    #e.g. nx.powerlaw_sequence(n) or cumulative
    #Others in: https://networkx.github.io/documentation/stable/reference/utils.html#module-networkx.utils.random_sequence 
G = nx.configuration_model(sequence) #standard = multigraph
