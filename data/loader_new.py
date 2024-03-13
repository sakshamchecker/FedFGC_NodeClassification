import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from privacy.coarsening import coarsen_a_dataset
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

def load_clients_data(data_name, client_number, tr_ratio, cr=False, cr_ratio=0):
    cora_dataset = Planetoid(root='data', name='Cora')
    data = cora_dataset[0]

    train_idx, test_idx = train_test_split(torch.arange(data.num_nodes), test_size=tr_ratio, random_state=42)
    train_idx = train_idx.tolist()
    test_idx = test_idx.tolist()

    # Index the Data object attributes with Python lists
    train_data = Data(x=data.x[train_idx], edge_index=data.edge_index, y=data.y[train_idx])
    test_data = Data(x=data.x[test_idx], edge_index=data.edge_index, y=data.y[test_idx])
    # Step 2: Define the number of clients and split the nodes
    num_clients = client_number
    num_nodes = train_data.num_nodes
    node_indices = torch.randperm(num_nodes)
    split_sizes = [num_nodes // num_clients] * num_clients
    split_sizes[-1] += num_nodes % num_clients  # Adjust for uneven split

    # Step 3: Create subgraphs for each client
    client_subgraphs = []
    start_idx = 0
    for size in split_sizes:
        end_idx = start_idx + size
        client_nodes = node_indices[start_idx:end_idx]
        client_subgraph = data.subgraph(client_nodes)
        client_subgraphs.append(client_subgraph)
        start_idx = end_idx

    # Step 4: Package subgraphs into client-specific datasets
    client_datasets = [Data(x=subgraph.x, edge_index=subgraph.edge_index, y=subgraph.y) for subgraph in client_subgraphs]

    return client_datasets, test_data, cora_dataset.num_features, cora_dataset.num_classes
