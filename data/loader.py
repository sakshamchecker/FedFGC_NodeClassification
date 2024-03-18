from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
# from privacy.coarsening import coarsening
from torch_geometric.utils import to_dense_adj,to_edge_index, from_scipy_sparse_matrix
import scipy.sparse as scpy
import numpy as np
# from partitioner import create_non_uniform_split
import numpy as np
batch_size=64
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

np.random.seed(42)
torch.manual_seed(42)
def split_graphs(n_clients, train_ratio, coarsen, data_name, alpha, cr_ratio,idxs=None):
  dataset=TUDataset(root='data/TU', name=data_name)
#   dataset = TUDataset(root='data/TU', name='PROTIENS')
  if data_name=='MUTAG':
      coarsen_params=[0.1,1,50,50]
  elif data_name=='PTC_MR': #0.01-10-0.1-1
      coarsen_params=[0.01,10,0.1,1]
  elif data_name=='NCI1':
      coarsen_params=[0.01,0.1,1,0.01]
  elif data_name=='PROTEINS': #0.01-0.1-0.01-1
       coarsen_params=[0.01,0.1,0.01,1]
  elif data_name=='AIDS': #0.01-0.01-0.01-0.01
        coarsen_params=[0.01,0.01,0.01,0.01]
  elif data_name=='ENZYMES': #0.01-0.01-0.01-0.1
        coarsen_params=[0.01,0.01,0.01,0.1]
  num_node_features = dataset.num_node_features
  num_classes = dataset.num_classes
  pivot = int(len(dataset)*train_ratio)
  #shuffle dataset
  dataset=dataset.shuffle()
  # val_set=dataset[pivot:]
  # train_set=dataset[:pivot]
  num_samples = len(dataset)
  if idxs is None:
    idxs=create_non_uniform_split(alpha=alpha, idxs=list(range(num_samples)), client_number=n_clients, is_train=True)
  client_train_loaders = []
  client_val_loaders = []
  # val_set = DataLoader(val_set, batch_size=batch_size, shuffle=True)
  for i in range(n_clients):
      
      # print(start_index, end_index)

      client_train_dataset = dataset[idxs[i]]

      pivot=int(len(client_train_dataset)*train_ratio)
      print('-------------------------',pivot, len(client_train_dataset))
      val_set=client_train_dataset[pivot:]
      val_set=DataLoader(val_set, batch_size=batch_size, shuffle=True)
      client_train_dataset=client_train_dataset[:pivot]
      client_train_dataset = DataLoader(client_train_dataset, batch_size=batch_size, shuffle=True)
      if coarsen:
        training_graphs=[]
        for step, data in enumerate(client_train_dataset):
            graphs=data.to_data_list()
            for i in tqdm(range(len(graphs))):
                adj=to_dense_adj(graphs[i].edge_index)[0]
                X=graphs[i].x
                # adj1,X1=coarsening(adj,X,0.1,1,50,50)
                adj1,X1=coarsening(adj,X,coarsen_params[0],coarsen_params[1],coarsen_params[2],coarsen_params[3], cr_ratio)
                # print(adj1.shape)
                # print(X.shape,X1.shape)
                A=scpy.csr_matrix(adj1)
                temp=from_scipy_sparse_matrix(A)
                g=Data(x=X1,edge_index=temp[0],edge_attr=temp[1],y=graphs[i].y)
                graphs[i]=g
            # coarsened_batch=Batch
            final_batch=Batch.from_data_list(graphs)
            training_graphs.append(final_batch)
        training_graphs=DataLoader(training_graphs, batch_size=batch_size, shuffle=True)
        client_train_loaders.append(training_graphs)
        del training_graphs
      else:
        client_train_loaders.append(client_train_dataset)
      client_val_loaders.append(val_set)
      del client_train_dataset, val_set
  return client_train_loaders, client_val_loaders, num_node_features, num_classes, idxs


def load_graphs(train_ratio, coarsen, data_name):
  dataset=TUDataset(root='data/TU', name=data_name)
#   dataset = TUDataset(root='data/TU', name='PROTIENS')
  if data_name=='MUTAG':
      coarsen_params=[0.1,1,50,50]
  elif data_name=='PTC_MR': #0.01-10-0.1-1
      coarsen_params=[0.01,10,0.1,1]
  elif data_name=='NCI1':
      coarsen_params=[0.01,0.1,1,0.01]
  elif data_name=='PROTEINS': #0.01-0.1-0.01-1
       coarsen_params=[0.01,0.1,0.01,1]
  elif data_name=='AIDS': #0.01-0.01-0.01-0.01
        coarsen_params=[0.01,0.01,0.01,0.01]
  elif data_name=='ENZYMES': #0.01-0.01-0.01-0.1
        coarsen_params=[0.01,0.01,0.01,0.1]
  num_node_features = dataset.num_node_features
  num_classes = dataset.num_classes
  dataset=dataset.shuffle()
  pivot = int(len(dataset)*train_ratio)
  val_set=dataset[pivot:]
  train_set=dataset[:pivot]
  num_samples = len(train_set)
  val_set = DataLoader(val_set, batch_size=batch_size, shuffle=True)
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
  if coarsen:
    training_graphs=[]
    for step, data in enumerate(train_loader):
        graphs=data.to_data_list()
        for i in tqdm(range(len(graphs))):
            adj=to_dense_adj(graphs[i].edge_index)[0]
            X=graphs[i].x
            # adj1,X1=coarsening(adj,X,0.1,1,50,50)
            adj1,X1=coarsening(adj,X,coarsen_params[0],coarsen_params[1],coarsen_params[2],coarsen_params[3])
            # print(adj1.shape)
            # print(X.shape,X1.shape)
            A=scpy.csr_matrix(adj1)
            temp=from_scipy_sparse_matrix(A)
            g=Data(x=X1,edge_index=temp[0],edge_attr=temp[1],y=graphs[i].y)
            graphs[i]=g
        final_batch=Batch.from_data_list(graphs)
        training_graphs.append(final_batch)
    train_loader=DataLoader(training_graphs, batch_size=batch_size, shuffle=True)
  return train_loader, val_set, num_node_features, num_classes


import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import to_networkx
#moleculenet from torchgeometric dataset
from torch_geometric.datasets import MoleculeNet
def coarsen_dataset(g_list, coarsen_params, cr_ratio):
    g=0
    g_mat=[]
    Node_tag_list=[]
    g_list2=[]

    while g < len(g_list):
        adj_matrix=nx.adjacency_matrix(g_list[g].g).toarray()
        adj_matrix=torch.Tensor(adj_matrix)
        # print(adj_matrix)
        X=np.array(g_list[g].node_tags).reshape(len(g_list[g].node_tags),1)
        X=torch.tensor(X)
        adj_matrix2,X2=coarsening(adj_matrix,X,coarsen_params[0],coarsen_params[1],coarsen_params[2],coarsen_params[3], cr_ratio)
        if adj_matrix2.shape!=adj_matrix.shape:
            
            adj_matrix2=adj_matrix2.numpy()
            grap=nx.from_numpy_array(adj_matrix2)
            g_mat.append(grap)
            X2=X2.reshape(1,len(X2))
            X2=X2.tolist()
            Node_tag_list.append(X2[0])
            
            g_list2.append(S2VGraph(grap,g_list[g].label,X2[0]))
            g+=1
    return g_list2

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag, coarsen=False, cr_ratio=0.5):
    # Download Mutag dataset using TUDataset
    if dataset=='MUTAG':
        coarsen_params=[0.1,1,50,50]
    elif dataset=='PTC_MR': #0.01-10-0.1-1
        coarsen_params=[0.01,10,0.1,1]
    elif dataset=='NCI1':
        coarsen_params=[0.01,0.1,1,0.01]
    elif dataset=='PROTEINS': #0.01-0.1-0.01-1
        coarsen_params=[0.01,0.1,0.01,1]
    elif dataset=='AIDS': #0.01-0.01-0.01-0.01
            coarsen_params=[0.01,0.01,0.01,0.01]
    elif dataset=='ENZYMES': #0.01-0.01-0.01-0.1
            coarsen_params=[0.01,0.01,0.01,0.1]
    elif dataset=='COLLAB': #0.01 1 50 50
            coarsen_params=[0.01,1,50,50]
    elif dataset=='IMDBBINARY': #0.01 0.01 0.01 0.01
            coarsen_params=[0.01,0.01,0.01,0.01]
    elif dataset=='IMDBMULTI': #0.01 0.01 0.01 0.01
            coarsen_params=[0.01,0.01,0.01,0.01]
    elif dataset=='REDDITBINARY': #0.01 0.01 0.01 0.01
            coarsen_params=[0.01,0.01,0.01,0.01]
    elif dataset=='Tox21_AhR_training': #0.01 0.01 0.01 0.01
            coarsen_params=[0.01,0.01,0.01,0.01]
    else:
        coarsen_params=[0.01,0.01,0.01,0.01]
    if dataset in ['MUTAG', 'PTC_MR', 'NCI1', 'PROTEINS', 'ENZYMES', 'COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'REDDITBINARY', 'Tox21_AhR_training', 'AIDS']:
        dataset = TUDataset(root='data', name=dataset)
    elif dataset in ['ESOL', 'FreeSolv', 'Lipophilicity', 'PCBA', 'MUV', 'HIV', 'BACE', 'BBBP', 'Tox21', 'ToxCast', 'SIDER', 'ClinTox']:
        dataset = MoleculeNet(root='data', name=dataset)
        original_tensor=dataset.y

        original_tensor[torch.isnan(original_tensor)] = 0

    # Create a dictionary to map unique rows to labels
        unique_rows = {tuple(row.numpy()): label for label, row in enumerate(torch.unique(original_tensor, dim=0))}

        # Convert to labeled format as a list of lists
        labeled_list_of_lists = [torch.tensor([unique_rows[tuple(row.numpy())]]) for row in original_tensor]
        dataset.y=torch.tensor(labeled_list_of_lists)

    print(dataset)
    
    # Extract data from TUDataset
    g_list = []
    label_dict = {}
    for i, data in enumerate(dataset):
        # Convert graph data to NetworkX format
        g = to_networkx(data, to_undirected=True)
        data.y=dataset.y[i]
        # Extract label from dataset
        label = int(dataset.y[i].item())

        # Add label to label dictionary
        if label not in label_dict:
            label_dict[label] = len(label_dict)

        # Extract node tags (degrees) if required
        if degree_as_tag:
            node_tags = [deg for _, deg in g.degree()]
        else:
            node_tags = None

        # Create S2VGraph object and append to list
        s2v_graph = S2VGraph(g, label, node_tags)
        g_list.append(s2v_graph)

    print(len(g_list))
    if coarsen:
      g_list=coarsen_dataset(g_list, coarsen_params, cr_ratio)
    ctr=0
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        if(len(edges)!=0):
            g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    remove=[]
    for g in range(len(g_list)):
        if isinstance(g_list[g].edge_mat,int):
            remove.append(g)

    for i in range(len(remove)):
        g_list.pop(remove[i]-i)


    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)
def load_data_pre(dataset, degree_as_tag,coarsen=False, cr_ratio=0.5):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}
    if dataset=='MUTAG':
      coarsen_params=[0.1,1,50,50]
    elif dataset=='PTC': #0.01-10-0.1-1
        coarsen_params=[0.01,10,0.1,1]
    elif dataset=='NCI1':
        coarsen_params=[0.01,0.1,1,0.01]
    elif dataset=='PROTEINS': #0.01-0.1-0.01-1
        coarsen_params=[0.01,0.1,0.01,1]
    elif dataset=='AIDS': #0.01-0.01-0.01-0.01
            coarsen_params=[0.01,0.01,0.01,0.01]
    elif dataset=='ENZYMES': #0.01-0.01-0.01-0.1
            coarsen_params=[0.01,0.01,0.01,0.1]
    elif dataset=='COLLAB': #0.01 1 50 50 
            coarsen_params=[0.01,1,50,50]
    elif dataset=='IMDBBINARY': #0.01 0.01 0.01 0.01
            coarsen_params=[0.01,0.01,0.01,0.01]
    elif dataset=='IMDBMULTI': #0.01 0.01 0.01 0.01
            coarsen_params=[0.01,0.01,0.01,0.01]
    elif dataset=='REDDITBINARY': #0.01 0.01 0.01 0.01
            coarsen_params=[0.01,0.01,0.01,0.01]
    elif dataset=='Tox21_AhR_training': #0.01 0.01 0.01 0.01
            coarsen_params=[0.01,0.01,0.01,0.01]
    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))
    if coarsen:
        g_list=coarsen_dataset(g_list, coarsen_params, cr_ratio)
    #add labels and edge_mat
    print(len(g_list))
    ctr=0       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        if(len(edges)!=0):
            g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    remove=[]
    for g in range(len(g_list)):
        if isinstance(g_list[g].edge_mat,int):
            remove.append(g)
            
    for i in range(len(remove)):
        g_list.pop(remove[i]-i)


    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())
    
    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def separate_data(graph_list, seed, fold_idx=1):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list



def load_clients_data(dataset, clients, alpha, coarsen=False, idxs=None, cr_ratio=0.5):
    if dataset in ['NCI1']:
        graph_list, num_classes = load_data_pre(dataset, True, coarsen, cr_ratio)
    else:
        graph_list, num_classes = load_data(dataset, True, coarsen, cr_ratio)
    if idxs is None:
        idxs=create_non_uniform_split(alpha=alpha, idxs=list(range(len(graph_list))), client_number=clients, is_train=True)
    for i in range(len(idxs)):
        idxs[i]=torch.tensor(idxs[i])
    client_graph_list=[]
    client_graph_list_test=[]
    for i in range(clients):
        # curr_list=[graph_list[j] for j in idxs[i]]
        curr_list=[]
        for j in idxs[i]:
            if j<len(graph_list):
                curr_list.append(graph_list[j])
        curr_train, curr_test=separate_data(curr_list, 42)
        client_graph_list.append(curr_train)
        client_graph_list_test.append(curr_test)
    return client_graph_list, client_graph_list_test, num_classes, idxs

