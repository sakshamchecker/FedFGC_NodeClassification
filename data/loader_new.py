import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from privacy.coarsening import coarse, preprocess


import scanpy as sc
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.decomposition import PCA
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix
from torch_geometric.utils import from_networkx
import numpy as np
from sklearn.neighbors import kneighbors_graph
import time
import networkx as nx
import sys


def partition_class_samples_with_dirichlet_distribution(
    N, alpha, client_num, idx_batch, idx_k
):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.000432batch_size], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))
    returning_prop=np.copy(proportions)
    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
        [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [
        idx_j + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size
def create_non_uniform_split(alpha, idxs, client_number, is_train=True):
    N = len(idxs)
    idx_batch_per_client = [[] for _ in range(client_number)]
    (
        idx_batch_per_client,
        min_size,
    ) = partition_class_samples_with_dirichlet_distribution(
        N, alpha, client_number, idx_batch_per_client, idxs
    )
    sample_num_distribution = []

    for client_id in range(client_number):
        sample_num_distribution.append(len(idx_batch_per_client[client_id]))
        

    return idx_batch_per_client

def load_base_data(data_name):
    dataset=TUDataset(root='data/TUDataset', name=data_name)
    return dataset

def load_central_data(data_name, batch_size, tr_ratio=0.8, cr=False, cr_ratio=0.5):
    dataset=load_base_data(data_name)
    dataset = dataset.shuffle()
    num_node_features=dataset.num_node_features
    num_classes=dataset.num_classes
    train_dataset=dataset[:int(len(dataset)*tr_ratio)]
    test_dataset=dataset[int(len(dataset)*tr_ratio):]
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    if cr:
        train_loader=coarsen_a_dataset(train_loader, coarsen_params=[0.1,0.01,1,1], batch_size=batch_size, cr_ratio=cr_ratio)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    del dataset, train_dataset, test_dataset
    return train_loader, test_loader, num_node_features, num_classes

# def load_clients_data(data_name, batch_size, client_number, tr_ratio=0.8, alpha=10,cr=False, cr_ratio=0.5, client_data=None):
#     dataset=load_base_data(data_name)
#     dataset = dataset.shuffle()
#     if not client_data:
#         client_data=create_non_uniform_split(alpha, list(range(len(dataset))), client_number)

#     client_train_loaders=[]
#     client_test_loaders=[]
#     test_idxs=[]
#     num_node_features=dataset.num_node_features
#     num_classes=dataset.num_classes
#     for i in range(client_number):
#         client=dataset[client_data[i]]
#         train_client=client[:int(len(client_data[i])*tr_ratio)]
#         test_client=client[int(len(client_data[i])*tr_ratio):]
#         test_idxs+=list(client_data[i][int(len(client_data[i])*tr_ratio):])
#         # test_client=client[int(len(client)*tr_ratio):]
#         train_loader = DataLoader(train_client, batch_size=batch_size, shuffle=True)
#         if cr:
#             train_loader=coarsen_a_dataset(train_loader,  coarsen_params=[0.01,0.01,0.01,0.01], batch_size=batch_size, cr_ratio=cr_ratio)
#         client_train_loaders.append(train_loader)
#         test_loader = DataLoader(test_client, batch_size=batch_size, shuffle=False)
#         client_test_loaders.append(test_loader)
#         del test_loader, train_loader, client, train_client, test_client
#     test_client=dataset[test_idxs]
#     test_loader = DataLoader(test_client, batch_size=batch_size, shuffle=False)
#     del test_client
#     return client_train_loaders, client_test_loaders,test_loader, num_node_features, num_classes, client_data

# def load_clients_data(data_name, batch_size, client_number, tr_ratio=0.8, alpha=10,cr=False, cr_ratio=0.5, client_data=None):
#     dataset=load_base_data(data_name)
#     dataset = dataset.shuffle()
#     train_dataset=dataset[:int(len(dataset)*tr_ratio)]
#     test_dataset=dataset[int(len(dataset)*tr_ratio):]
#     if not client_data:
#         client_data=create_non_uniform_split(alpha, list(range(len(train_dataset))), client_number)
#     client_train_loaders=[]
#     client_test_loaders=[]
#     test_idxs=[]
#     num_node_features=dataset.num_node_features
#     num_classes=dataset.num_classes
#     for i in range(client_number):
#         client=train_dataset[client_data[i]]
#         train_loader = DataLoader(client, batch_size=batch_size, shuffle=True)
#         if cr:
#             train_loader=coarsen_a_dataset(train_loader,  coarsen_params=[0.1,0.01,1,1], batch_size=batch_size, cr_ratio=cr_ratio)
#         client_train_loaders.append(train_loader)

#     train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     return client_train_loaders, train_loader, num_node_features, num_classes, client_data    
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
import os
import pickle
import random
def train_test_split_a_graph(data, tr_ratio):
    label_idx = data.y.numpy().tolist()
    print("label_idx", len(label_idx))
    train_idxs = []
    test_idxs = []
    li=[]
    for i in data.y:
        if i not in li:
            li.append(i)
    num_classes=len(li)
    for c in li:
        idx = (data.y==c).nonzero().view(-1)
        random.shuffle(idx)
        train_nodes = idx[:int(idx.shape[0]*tr_ratio)]
        test_nodes = idx[int(idx.shape[0]*tr_ratio):]
        train_idxs+=train_nodes
        test_idxs+=test_nodes
    train_data=data.subgraph(torch.tensor(train_idxs))
    test_data=data.subgraph(torch.tensor(test_idxs))

    # train_data = Data(x=data.x[train_idxs], edge_index=data.edge_index, y=data.y[train_idxs])
    # test_data = Data(x=data.x[test_idxs], edge_index=data.edge_index, y=data.y[test_idxs])
    return train_data, test_data
def shadow_target_split(data,tr_ratio, sh_ratio):
  target_idx, shadow_idx = train_test_split(torch.arange(data.num_nodes), test_size=sh_ratio, random_state=42)
  # target_idx = target_idx.to_list()
  # shadow_idx = shadow_idx.to_list()
  target_train, target_test= train_test_split_a_graph(data.subgraph(target_idx), tr_ratio)
  shadow_train, shadow_test= train_test_split_a_graph(data.subgraph(shadow_idx), tr_ratio)
  return target_train, target_test, shadow_train, shadow_test
def shadow_target_split_fl(data,tr_ratio, sh_ratio):
  target_idx, shadow_idx = train_test_split(torch.arange(data.num_nodes), test_size=sh_ratio, random_state=42)
  # target_idx = target_idx.to_list()
  # shadow_idx = shadow_idx.to_list()
#   target_train, target_test= train_test_split_a_graph(data.subgraph(target_idx), tr_ratio)
  target=data.subgraph(target_idx)
  shadow_train, shadow_test= train_test_split_a_graph(data.subgraph(shadow_idx), tr_ratio)
  return target, shadow_train, shadow_test
def clients_split_a_graph(data, num_clients, tr_ratio, cr, cr_ratio, cr_params):
    label_idx = data.y.numpy().tolist()
    print("label_idx", len(label_idx))
    train_idxs = [[] for i in range(num_clients)]
    test_idxs = []
    li=[]
    for i in data.y:
        if i not in li:
            li.append(i)
    num_classes=len(li)
    for c in li:
        idx = (data.y==c).nonzero().view(-1)
        random.shuffle(idx)
        train_nodes = idx[:int(idx.shape[0]*tr_ratio)]
        test_nodes = idx[int(idx.shape[0]*tr_ratio):]
        num_nodes=len(train_nodes)
        split_sizes = [num_nodes // num_clients] * num_clients
        split_sizes[-1] += num_nodes % num_clients  # Adjust for uneven split
        start_idx=0
        for i,size in enumerate(split_sizes):
          end_idx = start_idx + size
          client_nodes=train_nodes[start_idx:end_idx]
          train_idxs[i]+=client_nodes
          start_idx=end_idx
        test_idxs+=test_nodes
    # train_data=data.subgraph(torch.tensor(train_idxs))
    train_datasets=[]
    train_datasets_cr=[]
    for i in train_idxs:
      train_datasets.append(data.subgraph(torch.tensor(i)))
      train_data=data.subgraph(torch.tensor(i))
      if cr:
        X,adj,labels,features,NO_OF_CLASSES= preprocess(train_data.x, train_data.edge_index, train_data.y)
        X_new, edge_idx, labels_new=coarse(X=X, adj=adj, labels=labels, features=features, cr_ratio=cr_ratio,c_param=cr_params)
        train_datasets_cr.append(Data(x=X_new, edge_index=edge_idx, y=labels_new))
      else:
        train_datasets_cr=[[] for i in range(num_clients)]
          
    test_data=data.subgraph(torch.tensor(test_idxs))

    # train_data = Data(x=data.x[train_idxs], edge_index=data.edge_index, y=data.y[train_idxs])
    # test_data = Data(x=data.x[test_idxs], edge_index=data.edge_index, y=data.y[test_idxs])
    return train_datasets_cr, train_datasets, test_data
def load_clients_data(data_name, client_number, tr_ratio, cr=False, cr_ratio=0, sh_ratio=0):
    if data_name=='Cora':

        cora_dataset = Planetoid(root='data', name=data_name)
        c_params=[0.001, 0.0001, 1, 0.0001] 
        data = cora_dataset[0]
    elif data_name=='Citeseer':

        cora_dataset = Planetoid(root='data', name=data_name)
        c_params=[0.001, 0.001, 0.01, 0.01] 
        data = cora_dataset[0]
        # print(data)

    elif data_name == 'XIN':
        data_folder = 'data/Xin/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.1, 0.1, 1]

    elif data_name == 'baron_mouse':
        data_folder = 'data/Baron_Mouse/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.01, 0.01, 0.01]

    elif data_name == 'baron_human':
        data_folder = 'data/Baron_Human/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.01, 0.01, 0.01]

    elif data_name == 'Segerstolpe':
        data_folder = 'data/Segerstolpe/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.01, 0.01, 0.01]

    elif data_name == 'AMB':
        data_folder = 'data/AMB/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            cell_type_ = cell_type_['Class']
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.01, 0.01, 0.01]
    elif data_name == 'TM':
        data_folder = 'data/TM/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            cell_type_ = cell_type_['Class']
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.01, 0.01, 0.01]
    elif data_name=='Zheng':
        data_folder = 'data/Zheng/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            cell_type_ = cell_type_['Class']
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.01, 0.01, 0.01]
    num_features=data.x.shape[1]
    li=[]
    for i in data.y:
        if i not in li:
            li.append(i)
    num_classes=len(li)
    # train_idx, test_idx = train_test_split(torch.arange(data.num_nodes), test_size=tr_ratio, random_state=42)
    # train_idx = train_idx.tolist()
    # test_idx = test_idx.tolist()

    # # Index the Data object attributes with Python lists
    # train_data = Data(x=data.x[train_idx], edge_index=data.edge_index, y=data.y[train_idx])
    # test_data = Data(x=data.x[test_idx], edge_index=data.edge_index, y=data.y[test_idx])
    # if cr:
    #     X,adj,labels,features,NO_OF_CLASSES= preprocess(train_data.x, train_data.edge_index, train_data.y)
    #     X_new, edge_idx, labels_new=coarse(X=X, adj=adj, labels=labels, features=features, cr_ratio=cr_ratio,c_param=c_params)
    #     train_data=Data(x=X_new, edge_index=edge_idx, y=labels_new)
    # # Step 2: Define the number of clients and split the nodes
    # num_clients = client_number
    # num_nodes = train_data.num_nodes
    # node_indices = torch.randperm(num_nodes)
    # split_sizes = [num_nodes // num_clients] * num_clients
    # split_sizes[-1] += num_nodes % num_clients  # Adjust for uneven split

    # # Step 3: Create subgraphs for each client
    # client_subgraphs = []
    # start_idx = 0
    # for size in split_sizes:
    #     end_idx = start_idx + size
    #     client_nodes = node_indices[start_idx:end_idx]
    #     client_subgraph = data.subgraph(client_nodes)
        
    #     client_subgraphs.append(client_subgraph)
    #     start_idx = end_idx
    # test_nodes= test_data.num_nodes
    # node_indices = torch.randperm(test_nodes)
    # test_sub=data.subgraph(node_indices)
    # test_data = Data(x=test_sub.x, edge_index=test_sub.edge_index, y=test_sub.y)
    # # Step 4: Package subgraphs into client-specific datasets
    # client_datasets = [Data(x=subgraph.x, edge_index=subgraph.edge_index, y=subgraph.y) for subgraph in client_subgraphs]
    target_data, shadow_train, shadow_test = shadow_target_split_fl(data, tr_ratio=tr_ratio, sh_ratio=sh_ratio)
    client_datasets_cr, client_datasets, test_data = clients_split_a_graph(data=target_data, num_clients=client_number, tr_ratio=tr_ratio, cr=cr, cr_ratio=cr_ratio, cr_params=c_params)
    return client_datasets_cr, client_datasets, test_data,shadow_train, shadow_test, num_features, num_classes



def perform_qc(adata):
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    min_genes_by_cells = 200
    adata = adata[adata.obs.n_genes_by_counts > min_genes_by_cells, :]

    min_cells_by_genes = 3
    adata = adata[:, adata.var.n_cells_by_counts > min_cells_by_genes]

    # Filter based on total_counts or pct_dropout_by_counts

    return adata

def Graph_Build_from_data(rna_matrix_,cell_type):

# Assuming 'y_train' and 'y_test' are your original labels with strings
    label_encoder = LabelEncoder()
    cell_type = label_encoder.fit_transform(cell_type)
    try:
        rna_matrix_ = rna_matrix_.set_index(['Unnamed: 0'])
    except:
        print('')
    adata = sc.AnnData(rna_matrix_)


    rna_matrix_qc = perform_qc(adata)
    data = rna_matrix_qc.X
    pca = PCA(n_components=50)
    data = pca.fit_transform(data)


    knn = NearestNeighbors(n_neighbors=5)  # Adjust 'n_neighbors' as needed
    knn.fit(data)

    # Build the KNN graph
    knn_graph = knn.kneighbors_graph(data, mode='distance')
    G = nx.from_scipy_sparse_array(knn_graph)
    obj = from_networkx(G)
    obj.x = torch.Tensor(data)
    obj.y = torch.Tensor(cell_type)
    return obj
def one_hot(x):
    data=x
    unique_values = np.unique(data)

# Create a dictionary mapping each unique value to an integer
    value_to_int = {value: i for i, value in enumerate(unique_values)}

    # Convert the string array to numerical array using the mapping
    numerical_data = np.array([value_to_int[value] for value in data])

    # Determine the number of unique values in the numerical array
    num_classes = len(unique_values)

    # Initialize an empty array to hold the one-hot encoded data
    one_hot_encoded = np.zeros((len(data), num_classes))

    # Iterate over each value in the numerical array
    for i, val in enumerate(numerical_data):
        # Set the corresponding element in the one-hot encoded array to 1
        one_hot_encoded[i, val] = 1
    return torch.tensor(one_hot_encoded)
def load_central_data(data_name, tr_ratio, cr=False, cr_ratio=0, sh_ratio=0.5):
    if data_name=='Cora':

        cora_dataset = Planetoid(root='data', name=data_name)
        c_params=[0.001, 0.0001, 1, 0.0001] 
        data = cora_dataset[0]
        # print(data)
    elif data_name=='Citeseer':

        cora_dataset = Planetoid(root='data', name=data_name)
        c_params=[0.001, 0.001, 0.01, 0.01] 
        data = cora_dataset[0]
        # print(data)
    

    elif data_name == 'XIN':
        data_folder = 'data/Xin/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.1, 0.1, 1]

    elif data_name == 'baron_mouse':
        data_folder = 'data/Baron_Mouse/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.01, 0.01, 0.01]

    elif data_name == 'baron_human':
        data_folder = 'data/Baron_Human/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.01, 0.01, 0.01]

    elif data_name == 'Segerstolpe':
        data_folder = 'data/Segerstolpe/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.01, 0.01, 0.01]

    elif data_name == 'AMB':
        data_folder = 'data/AMB/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            cell_type_ = cell_type_['Class']
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.01, 0.01, 0.01]
    elif data_name == 'TM':
        data_folder = 'data/TM/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            cell_type_ = cell_type_['Class']
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.01, 0.01, 0.01]
    elif data_name=='Zheng':
        data_folder = 'data/Zheng/'
        if os.path.isfile(data_folder + 'processed.pkl'):
            with open(data_folder + 'processed.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            rna_matrix_ = pd.read_csv(data_folder + 'data.csv')
            cell_type_ = pd.read_csv(data_folder + 'Labels.csv')
            cell_type_ = cell_type_['Class']
            data = Graph_Build_from_data(rna_matrix_, cell_type_)
            with open(data_folder + 'processed.pkl', 'wb') as f:
                pickle.dump(data, f)
        c_params = [0.01, 0.01, 0.01, 0.01]
    num_features=data.x.shape[1]

    li=[]
    for i in data.y:
        if i not in li:
            li.append(i)
    num_classes=len(li)

    # train_idx, test_idx = train_test_split(torch.arange(data.num_nodes), test_size=tr_ratio, random_state=42)
    # train_idx = train_idx.tolist()
    # test_idx = test_idx.tolist()

    # # Index the Data object attributes with Python lists
    # train_data = Data(x=data.x[train_idx], edge_index=data.edge_index, y=data.y[train_idx])
    # test_data = Data(x=data.x[test_idx], edge_index=data.edge_index, y=data.y[test_idx])
    # train_nodes= train_data.num_nodes
    # node_indices = torch.randperm(train_nodes)
    # train_sub=data.subgraph(node_indices)
    # train_data = Data(x=train_sub.x, edge_index=train_sub.edge_index, y=train_sub.y)
    # test_nodes= test_data.num_nodes
    # node_indices = torch.randperm(test_nodes)
    # test_sub=data.subgraph(node_indices)
    # test_data = Data(x=test_sub.x, edge_index=test_sub.edge_index, y=test_sub.y)
    # train_data, test_data = train_test_split_a_graph(data=data,tr_ratio=tr_ratio)
    target_train, target_test, shadow_train, shadow_test = shadow_target_split(data, tr_ratio=tr_ratio, sh_ratio=sh_ratio)
    print(target_train)
    target_train_cr=None
    if cr:
        X,adj,labels,features,NO_OF_CLASSES= preprocess(target_train.x, target_train.edge_index, target_train.y)
        X_new, edge_idx, labels_new=coarse(X=X, adj=adj, labels=labels, features=features, cr_ratio=cr_ratio,c_param=c_params)
        target_train_cr=Data(x=X_new, edge_index=edge_idx, y=labels_new)
    return target_train,target_train_cr, target_test, shadow_train, shadow_test, num_features, num_classes