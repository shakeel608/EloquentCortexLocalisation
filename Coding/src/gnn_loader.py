"""Author: Dr. SHAKEEL AHMAD SHEIKH"""

#This is a GNN Data Loader using NetworkX and DGL Package 

import os 
import json
import torch
import networkx as nx
from dgl.data import DGLDataset
import dgl
from networkx.readwrite import json_graph

class GNNDataSet(DGLDataset):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self):
        super(GNNDataSet, self).__init__(name='gnn_dataset')
        self.modality = 't1'
        self.graphs = self.load() 




    def __getitem__(self, i):
        # get one example by index
        graph_nx = self.load_graph(self.graphs[i])
        print("Graph in get item===>\n",graph_nx.number_of_edges())
        #return self.graphs[i]
        import numpy as np
        #print("===#####",graph_nx.graph)
        #for nd, attrib in graph_nx.nodes(data=True):
            #print(f"{nd}: {attrib['features']}:{attrib['label']}")
            #print(f"{nd}: {attrib}")
        #print("NX G==>", graph_nx)
        #print("Adj of loaded graph", np.unique(nx.to_numpy_array(graph_nx)))
        #Converts graph from NetworkX to DGL Tensor
        dgl_graph = dgl.from_networkx(graph_nx, node_attrs=['features', 'label'])
        #print("DGL Node Data", dgl_graph.ndata)  #Prints node features
        print("DGL Node Features", dgl_graph.ndata['features'].shape, dgl_graph.ndata['label'].shape)  #Prints node features
        #print("DGL E data", dgl_graph.edata)  #Prints Edge features
        print("DGl Adj", dgl_graph.adjacency_matrix().to_dense().shape)
        print("Graph in DGL #Edges===>\n",dgl_graph.number_of_edges())
       
        print("Node Features===>", dgl_graph.ndata['features'].shape)
        print("Node Lables===>", dgl_graph.ndata['label'].shape)
        
        #Counter To Extract Majority Lable from Nodes for Each Graph
        from collections import Counter
        label_counts = Counter(dgl_graph.ndata['label'].tolist())
        majority_label, majority_count = label_counts.most_common()[0]
        print("Majority Label Counts", label_counts,majority_label)
        graph_label = majority_label
        #exit(0)
        #Manual
       # nn = graph_nx.number_of_nodes()
       # dgl_G = dgl.DGLGraph()
       # dgl_G.add_nodes(nn)
       # edges = list(graph_nx.edges())
       # src, dst = zip(*edges)
       # dgl_G.add_edges(src, dst)
       # # Print NetworkX and DGL graph edge counts
       # #for node in graph_nx:
       #     #print("Node in Nx",node)
       # for node in dgl_G.nodes():
       #     #print("Node=>",node)
       #     #dgl_G.nodes[node]['features'] = torch.tensor(graph_nx.nodes[int(node)]['features'], dtype=torch.float32)
       #     dgl_G.nodes[node]['features'] = torch.tensor(2, dtype=torch.float32)
       #     print("Node Features and Label",graph_nx.nodes[int(node)]['features'])
       #     #dgl_G.nodes[node]['label'] = torch.tensor(graph_nx.nodes[int(node)]['label'], dtype=torch.int32)
       # print("Number of Edges in NetworkX Graph:", graph_nx.number_of_edges())
       # print("Number of Edges in DGL Graph:", dgl_G.number_of_edges())
       # print("DGL Graph:", dgl_G)
        #exit(0)
        return dgl_graph, graph_label

    def __len__(self):
        # number of data examples
        return len(self.graphs)

    def load_graph(self, path):
        #Load graph
        with open(path, 'r') as f:
            graph_load = json.loads(f.read())
            return nx.readwrite.json_graph.node_link_graph(graph_load)


    def load(self):
        # load processed data from directory `self.save_path`
        print(os.getcwd())
        base_dir_name = "generated_graphs"
        #generated_graphs_multimodal_sv_300_K_11
        src_dir = base_dir_name +"_"+ self.modality + "_sv_nn_300_K_11"
        graphdata_dir = os.path.join(os.getcwd(), src_dir)  #Graphs are store in path as JSON Files
        print(graphdata_dir)
        items = os.listdir(graphdata_dir)
        print(len(items))
        graph_paths = []
        for item in items:
           #print(item)
           graph_paths.append(os.path.join(graphdata_dir, item))
           #print("Graph data paths",graph_paths)#;exit(0)
           


        return graph_paths


import torch.nn as nn
import torch.nn.functional as F

#Define Grah Convolutional Layer

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, graph, node_feats):
        graph.ndata['features'] = inp_feats
        graph.send()  #Send Message
        graph.receive()  #Receive Message


if __name__=="__main__":
    gnn_dataset = GNNDataSet()
    gnn_loader = dgl.dataloading.GraphDataLoader(dataset=gnn_dataset, batch_size=4) 

    for data, label in gnn_loader:
        feats = data.ndata['features']
        print("Features", feats.shape)
        print("data in dataloader *******>", data, label)
        #logits = model(feats, label)
        #loss = F.cross_entropy(logits, label)
        #optim.zero_grad()
        #loss.backward()
        #optim.step()
        break

