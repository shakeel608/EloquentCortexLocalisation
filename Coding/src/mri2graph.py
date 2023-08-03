#Author: Dr. Shakeel Ahmad Sheikh
#Affiliation: PostDoctoral Research Scientist, CITEC, University of Bielefeld, Germany
#Date: June 1, 2023
#Description: This script generates a connected graph with edges from 3D MRI image uisng SLIC alogrithm and saves them in the output directory.
#The SLIC algorith is used to generate super voxels referred to as nodes  

import numpy as np 
import scipy
import networkx as nx
from scipy.spatial.distance import cdist
from skimage.segmentation import slic
from scipy import ndimage
import nibabel as nib
import os
import networkx as nx
import json
import shutil
import matplotlib.pyplot as plt


class MRI2Graph():
    def __init__(self, num_modalities, mod_path, mu, sigma, num_nodes):
        print("Status::>")
        self.num_modalities = num_modalities
        self.mod_path = mod_path
        self.dataset_mu = mu
        self.dataset_sigma = sigma
        self.num_nodes = num_nodes   #approximate number of nodes
        self.quantile_range = [0.10, 0.25, 0.50, 0.75, 0.90]
        self.K = 11
    """Computes quantiles of range 'quantile_range' based on SLIC partititon""" 
    def compute_quantiles(self, x):
        temp = np.quantile(x, self.quantile_range)
        #print("Quantiles", temp, "\n")

        return temp 

    """Returns mode of the SLIC Centroids"""

    def compute_mode(self, x):
        val, counts = np.unique(x, return_counts=True)   #x is an array
        mode = counts.argmax()

        #print("Val = {} Counts = {}, Mode = {}, Val[Mode] = {}".format(val, counts, mode, val[mode])) #;exit(0)
        return val[mode]

    def normalize(self, img_array,flatten_img=True):
        #print("In Class MRI Normalise", img_array.shape)
        #print("length", len(img_array.shape))
        maxes = np.quantile(img_array, 0.995).astype(np.float32)  #99% quantile from full MRI Image SIngle Modality
        if maxes == 0:
            norm_img = img_array/(maxes + 1e-7)
            
        else:
            norm_img = img_array/maxes
        return norm_img


    #Extracts Super Voxel Partition = Number of Apprioximate Nodes in a Graph
    def get_super_voxels(self, mri_img):
        print("Num of Nodes==>{}, Img Shape===>{}".format(self.num_nodes, mri_img.shape))
        mri_img = mri_img.astype(np.float64)
        print("Lenth num _modalities", len(self.num_modalities))
        if len(self.num_modalities) > 2:
            #mri_img = mri_img.reshape(-1, mri_img.shape[-2], mri_img.shape[-1])
            print("Concatentated Shape", mri_img.shape)
            #Channel/Modality axis = 0
            super_voxels = slic(mri_img, n_segments=self.num_nodes,channel_axis=0, convert2lab=False)
            #print("super Voxels = Num Noes generated", super_voxels.shape, np.unique(super_voxels))
            num_super_voxels = np.max(super_voxels) + 1 #Because partitioning starts from 0
            
        else:
            super_voxels = slic(mri_img, n_segments=self.num_nodes,channel_axis=None, convert2lab=False)
            #print("super Voxels = Num Noes generated", super_voxels.shape, np.unique(super_voxels))
            num_super_voxels = np.max(super_voxels) + 1 #Because partitioning starts from 0

            #exit(0)

        return super_voxels.astype(np.int16)


    """This function extracts node features Shape is N X 5*M (N: Num of SuperVoxels/Nodes Extracted 
    M: Modalities, num of quantiles)
    and Node centroids with Shape N X 3 (3 for X,Y,Z)
    Centroids: Centre of masses grouped by Node ID
    """
    def extract_node_features(self, nodes, mri, num_nodes):
        if len(self.num_modalities) > 2:
            num_modalities = mri.shape[0]
            node_features = []
            for m in range(num_modalities):
                node_modality_feature = ndimage.labeled_comprehension(mri[m,:,:,:], labels=nodes, func=self.compute_quantiles, index=range(1, num_nodes), out_dtype='object',default=-1.0)
                node_modality_feature = np.stack(node_modality_feature, axis=0)
                print("Node modality feature", node_modality_feature.shape)#;exit(0)
                node_features.append(node_modality_feature)
            node_features = np.concatenate(node_features, axis=-1)   #Concatentaing Node features from multiple modalities 
            print("Combined Node Features", node_features.shape)#;exit(0)

        else:
            node_features = ndimage.labeled_comprehension(mri, labels=nodes, func=self.compute_quantiles, index=range(1, num_nodes), out_dtype='object', default=-1.0)
            node_features = np.stack(node_features, axis=0)
        print("Node Features", node_features.shape)

        node_centroids = np.array(ndimage.center_of_mass(np.ones(nodes.shape), nodes, range(1, num_nodes)))

        print("node_centroids",node_centroids.shape)
        #exit(0)
        return node_features, node_centroids


    


    def mri2graph(self):
        """This function first standarizes each MRI image and 
        then extracts super voxels and converts image to graph with node features"""
        
        print("Extrating Class  Graph Supervoxels from MRI Image")
        num_modalities = len(self.num_modalities)

        print("Num Modals", self.num_modalities)
        #For Multimodal
        if num_modalities > 2:  #1Modality + 1 Segmentation Label
            modality_path_t1 = self.mod_path[self.num_modalities[0].split(".")[0].split("_")[-1]]  # 0 for T1 Modality
            modality_path_t1ce = self.mod_path[self.num_modalities[1].split(".")[0].split("_")[-1]]
            modality_path_t2 = self.mod_path[self.num_modalities[2].split(".")[0].split("_")[-1]]
            modality_path_flair = self.mod_path[self.num_modalities[3].split(".")[0].split("_")[-1]]
            modality_path_seg = self.mod_path[self.num_modalities[4].split(".")[0].split("_")[-1]]
            mri_files = os.listdir(modality_path_t1)
            for i, mri in enumerate(mri_files):
                comb_mri_modalities = []
                #T1 Modality
                mri_norm = self.read_mri(mri, modality_path_t1, 0)
                stnd_mri = self.standardize_img(mri_norm, self.dataset_mu, self.dataset_sigma)
                comb_mri_modalities.append(stnd_mri)
                #T1CE Modality
                mri_norm = self.read_mri(mri, modality_path_t1ce, 1)
                stnd_mri = self.standardize_img(mri_norm, self.dataset_mu, self.dataset_sigma)
                comb_mri_modalities.append(stnd_mri)
                #T2 Modality
                mri_norm = self.read_mri(mri, modality_path_t2, 2)
                stnd_mri = self.standardize_img(mri_norm, self.dataset_mu, self.dataset_sigma)
                comb_mri_modalities.append(stnd_mri)
                #FLAIR Modality
                mri_norm = self.read_mri(mri, modality_path_flair, 3)
                stnd_mri = self.standardize_img(mri_norm, self.dataset_mu, self.dataset_sigma)
                comb_mri_modalities.append(stnd_mri)

                #Seg Labels
                mri_seg = self.read_mri(mri, modality_path_seg, 4)

                
                #Combine Image Modalities in Channel wise like RGB
                comb_mri_modalities = np.stack(comb_mri_modalities, axis=0)
                print("COMBINE", comb_mri_modalities.shape)

                super_voxels = self.get_super_voxels(comb_mri_modalities)
                #K = 11
                print("Modal length", comb_mri_modalities.shape)
                node_features, node_centroids  = self.extract_node_features(np.copy(super_voxels), comb_mri_modalities, self.num_nodes)
                adj_matrix = self.construct_adj_matrix(node_features, node_centroids, self.K)
                graph = nx.from_numpy_array(adj_matrix)
                print("Modality FIle", mri)
                """Extract Label for Node which is present in highest in SV partition"""
                supervoxel_labels = self.get_supervoxel_labels(mri_seg, super_voxels) 
                print("Super Voxel Labels", supervoxel_labels, np.unique(supervoxel_labels)) #; exit(0)
                #nx.draw_networkx(graph, with_labels=True, font_size=5, node_size=10)
                ##plt.savefig("MMGraphGeneratedER.pdf")
                self.save_graph(graph, node_features, supervoxel_labels, mri)
                #plt.show()
                #exit(0)
        
        #For Unimodal
        else:
            modality_path = self.mod_path[self.num_modalities[0].split(".")[0].split("_")[-1]]
            modality_path_seg = self.mod_path[self.num_modalities[1].split(".")[0].split("_")[-1]]
            mri_files = os.listdir(modality_path)
            for i, mri in enumerate(mri_files):
                mri_norm = self.read_mri(mri, modality_path, 0)
                mri_seg  = self.read_mri(mri, modality_path_seg, 1)
                stnd_mri = self.standardize_img(mri_norm, self.dataset_mu, self.dataset_sigma)
                #stnd_mri = mri_norm
                #Extract Super Voxels
                super_voxels = self.get_super_voxels(stnd_mri)
                node_features, node_centroids = self.extract_node_features(np.copy(super_voxels),stnd_mri, self.num_nodes)
                
                """Extract Label for Node which is present in highest in SV partition"""
                if np.any(np.isnan(mri_seg)):
                    print("Nans present in labels")
                    continue
                supervoxel_labels = self.get_supervoxel_labels(mri_seg, super_voxels) 
                print("Super Voxel Labels", supervoxel_labels, np.unique(supervoxel_labels)) #; exit(0)
                #K: Number of adjacent neighbors for each node, otherwise it will be a complete graph 
                #K = 11
                #if K:
                    #Call Adjacency Matric COnstruction function for Supervoxel/Nodes
                adj_matrix = self.construct_adj_matrix(node_features, node_centroids, self.K)

                graph = nx.from_numpy_array(adj_matrix)
                self.save_graph(graph, node_features, supervoxel_labels, mri)
                #exit(0)
                


    def get_supervoxel_labels(self, seg_labels, super_voxels):
        #Returns label for node /sv which is present in majority
        supervoxel_labels = ndimage.labeled_comprehension(seg_labels, labels=super_voxels, func=self.compute_mode, index=range(1, self.num_nodes),out_dtype='int32',default=0.0)                
        return supervoxel_labels



    #Load graph as nx variable
    def load_graph(self, path):
        #Load graph 
        with open(path, 'r') as f:
            graph_load = json.loads(f.read())
            return nx.readwrite.json_graph.node_link_graph(graph_load)

    
    def plot_graph(self, graph, g_name):
        print("Plotting a Graph Sample")
        nx.draw_networkx(graph, with_labels=True, font_size=5, node_size=10)
        plt.savefig(g_name)


    
    def save_graph(self, graph, node_features, supervoxel_labels, file_path):
        #Saves grah as pdf and as JSON file
        #print("Graph statistics==>",graph.number_of_nodes(), graph.number_of_edges())
        #self.plot_graph(graph, "G_saved.pdf")
        #Check if segment labels are provided 
        NODE_LABELS_PROVIDED = True if supervoxel_labels is not None else False

        for nd in graph.nodes:
            feats  = list(node_features[nd])
            graph.nodes[nd]["features"]  = feats
            #print("Saving nodes features", feats)
            #print("Saved nodes features", graph.nodes[nd])
            #print("\n\n")
            #Append Labels to each Node
            if NODE_LABELS_PROVIDED:
                node_label = int(supervoxel_labels[nd])
                graph.nodes[nd]['label'] = node_label

        dst_dir = "generated_graphs_t2_sv_300_K_"+str(self.K)
        #if os.path.exists(os.path.join(os.getcwd(), dst_dir)):
            #shutil.rmtree(dst_dir)
            #print("[Removing dst_dir from mri2graphy.py]")
        if not os.path.exists(os.path.join(os.getcwd(), dst_dir)):
            print("[Creating dst_dir from mri2graphy.py]")
            os.makedirs(dst_dir)

        if len(self.num_modalities) > 2: 
            save_path = dst_dir +"/" + file_path.split(".")[0].rpartition("_")[0] + "_multimodal.graph.json"
        else:
            save_path = dst_dir +"/" + file_path.split(".")[0] +".graph.json"

        print("file path", save_path)
        #Saving graph as JSON file
        graph_json = nx.readwrite.json_graph.node_link_data(graph)
        temp = json.dumps(graph_json)
        with open(save_path, 'w') as f:
            f.write(temp)
        print("file path saved", save_path)

        #G = self.load_graph(save_path)
        #print("G statistics==>",G.number_of_nodes(), G.number_of_edges())
        #self.plot_graph(G, "G_loaded.pdf")
        #print("Loaded Grapgh", G)
        #exit(0)

    #Generates Adjacency Matrix. Weighted and Enforces Regularity
    def construct_adj_matrix(self, node_features, node_centroids, K, weighted_adj=True, enforce_regularity = False, complete_adj_mat=True):
        """
        This function return adjacency matrix of the given super voxels/nodes
        """
        centroid_distances = cdist(node_centroids, node_centroids, metric='euclidean')
        print("Distances of Centroids SHape",centroid_distances.shape)
        adj_mat = np.zeros(centroid_distances.shape)
        
        #Solution 2 
        #If False, then all nodes will have exactly K neighbors but some nodes which are neighbors for other nodes can create incosistencies 
        #0 [[5 3 0 2 1 4]
        #1  [2 4 3 5 2 0]
        #2  [2 3 1 0 5 4] 
        #  ], The some nodes might be having only one directional nodes and adjacency matrix won't be consistent 
        #complete_adj_mat = True
        #enforce_regularity = False
        if enforce_regularity:
            top_k_nn = np.argsort(centroid_distances, axis=1)  #top_k_nn is Top K Nearest neighbor nodes
            for node in range(len(top_k_nn)):
                possible_nodeneighbors = top_k_nn[node][top_k_nn[node] > node]  #node: sorted array of nodes with distances w.r.t current node
                neighbors_needed = int(K - np.sum(adj_mat[node]))
                if neighbors_needed:
                    #print("node", node)
                    #print("Nn_needed", neighbors_needed)
                    neighbor_nodes_to_add = possible_nodeneighbors[:neighbors_needed]
                    #neighbor_nodes_to_add = neighbor_nodes_to_add.astype(np.int16)
                    #print("Nn_to add", neighbor_nodes_to_add)
                    adj_mat[node][neighbor_nodes_to_add] = 1.0
                    adj_mat[neighbor_nodes_to_add, node] = 1.0  #Transpose to make adj matrix consistent
        

        else:
            #What about loop Since adj will have 1's at diagonal
            #This ensures exactly K neighbors for each super voxe/node but based on the distance metric might have unidirections.
            #But we want undirected graph here that adjacency matrix should be symmetric. 
            #Solutions, Add 1's to [j,i] if [i,j] =1
            top_k_neighbors = np.argsort(centroid_distances, axis=1)[:,:K]
            for i in range(len(top_k_neighbors)):
                adj_mat[i][top_k_neighbors[i]]  = 1.0   #Placing 1 only at closeset nodes i.e neighbors
            #print("Adjancey Matrix", adj_mat.shape, adj_mat, np.count_nonzero(adj_mat == 1.0))
        
            #Solution 1, Enfore Regularity Add 1's to [j,i] if [i,j] =1
            if complete_adj_mat:
                for row in range(adj_mat.shape[0]):
                    for col in range(adj_mat.shape[1]):
                        if adj_mat[row, col] == 1.0:
                            adj_mat[col, row] = 1.0 
                        if adj_mat[col, row] == 1.0:
                            adj_mat[row, col] = 1.0



        #print("Symmetric Check Adj", np.array_equal(adj_mat, adj_mat.T))

        #Removing Loops
        np.fill_diagonal(adj_mat, 0.0)

        num_neighbors_sv = np.sum(adj_mat, axis=1)
        #import pandas as pd
        #df = pd.DataFrame({'K':num_neighbors_sv})
        #df.to_csv("K_NN", index=False)
        #print("DF", df)
        #print("Sum Nodes with greater than 10 neighbors",(num_neighbors_sv > 10).sum())
        #print("Adjancey Matrix",adj_mat, np.sum(adj_mat, axis=1))
        #exit(0)

        #Weighted Matrix
        if weighted_adj:
            #compute  euclidean distances between super voxels intensities i.e node features
            feat_dist = cdist(node_features, node_features, metric='euclidean')
            feat_dist = feat_dist/np.amax(feat_dist)
            sig = 0.5
            edge_wts = np.exp(-(feat_dist**2)/(2*sig**2))
            adj_mat = adj_mat * edge_wts
            return adj_mat
            print("Adjancey Matrix",adj_mat, np.sum(adj_mat, axis=1))
        else:
            return adj_mat

      

        


    def standardize_img(self, img_array, mean, std):
        centered = img_array - mean
        standardized = centered/std
        return standardized


    def read_mri(self, mri, modality_path, idx):
        """
        Reads and return normalised MRI image using Nibabel package
        idx : index for modality
        0: T1
        1: T1CE
        2: T2
        3: FLAIR
        4: Segmentation if multimodal else 1

        """
        img = mri.rpartition("_")[0] + self.num_modalities[idx]
        img_path = os.path.join(modality_path,img)
        mri_data = nib.load(img_path)
        mri_array = mri_data.get_fdata()
        mri_array = np.transpose(mri_array, (2,0,1))
        mri_norm = self.normalize(mri_array)
        return mri_norm
