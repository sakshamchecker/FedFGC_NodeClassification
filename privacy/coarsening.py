from scipy.sparse import random
from scipy.stats import rv_continuous
from scipy import sparse
from scipy.sparse import csr_matrix
from random import sample
from torch_geometric.utils import to_dense_adj
import torch
import numpy as np
#import tqdm 
from tqdm import tqdm
def convertScipyToTensor(coo):
  try:
    coo = coo.tocoo()
  except:
    coo = coo
  values = coo.data
  indices = np.vstack((coo.row, coo.col))

  i = torch.LongTensor(indices)
  v = torch.FloatTensor(values)
  shape = coo.shape

  return torch.sparse.FloatTensor(i, v, torch.Size(shape))

class CustomDistribution(rv_continuous):
    def _rvs(self,  size=None, random_state=None):
        return random_state.standard_normal(size)
def get_laplacian(adj):
    b=torch.ones(adj.shape[0])
    return torch.diag(adj@b)-adj

def one_hot(x, class_count):
    xtemp = x
    # return torch.eye(class_count)[xtemp, :].cuda()
    return torch.eye(class_count)[xtemp, :]
def experiment(lambda_param,beta_param,alpha_param,gamma_param,C,X_tilde,theta,X,p,n,k):
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      thresh = 1e-10
      ones = csr_matrix(np.ones((k,k)))
      ones = convertScipyToTensor(ones)
      ones = ones.to_dense()
      J = np.outer(np.ones(k), np.ones(k))/k
      J = csr_matrix(J)
      J = convertScipyToTensor(J)
      J = J.to_dense()
      zeros = csr_matrix(np.zeros((p,k)))
      zeros = convertScipyToTensor(zeros)
      zeros = zeros.to_dense()
      X_tilde = convertScipyToTensor(X_tilde)
      X_tilde = X_tilde.to_dense()
      C = convertScipyToTensor(C)
      C = C.to_dense()
      eye = torch.eye(k)
      L=None
      try:
        theta = convertScipyToTensor(theta)
      except:
        theta = theta
      try:
        X = convertScipyToTensor(X)
        X = X.to_dense()
      except:
        X = X

      if(torch.cuda.is_available()):
        print("yes")
        X_tilde = X_tilde.cuda()
        C = C.cuda()
        theta = theta.cuda()
        X = X.cuda()
        J = J.cuda()
        zeros = zeros.cuda()
        ones = ones.cuda()
        eye = eye.cuda()

      def update(X_tilde,C,i,L):
          # global L
          thetaC = theta@C
          CT = torch.transpose(C,0,1)
          X_tildeT = torch.transpose(X_tilde,0,1)
          CX_tilde = C@X_tilde
          t1 = CT@thetaC + J
          term_bracket = torch.linalg.pinv(t1)
          thetacX_tilde = thetaC@(X_tilde)

          L = 1/k

          t1 = -2*gamma_param*(thetaC@term_bracket)
          t2 = alpha_param*(CX_tilde-X)@(X_tildeT)
          t3 = 2*thetacX_tilde@(X_tildeT)
          t4 = lambda_param*(C@ones)
          t5 = 2*beta_param*(thetaC@CT@thetaC)
          T2 = (t1+t2+t3+t4+t5)/L
          Cnew = (C-T2).maximum(zeros)
          t1 = CT@thetaC*(2/alpha_param)
          t2 = CT@C
          t1 = torch.linalg.pinv(t1+t2)
          t1 = t1@CT
          t1 = t1@X
          X_tilde_new = t1
          Cnew[Cnew<thresh] = thresh
          for i in range(len(Cnew)):
              Cnew[i] = Cnew[i]/torch.linalg.norm(Cnew[i],1)
          for i in range(len(X_tilde_new)):
            X_tilde_new[i] = X_tilde_new[i]/torch.linalg.norm(X_tilde_new[i],1)
          return X_tilde_new,Cnew, L


      for i in tqdm(range(30)):
          X_tilde,C, L = update(X_tilde,C,i, L)

      return X_tilde,C


def coarse(X,adj,labels,features, cr_ratio,c_param):
  NO_OF_CLASSES =  len(set(np.array(labels)))
  p = X.shape[0]
  k = int(p*cr_ratio)   ###Corasening ratio
  n = X.shape[1]
  temp = CustomDistribution(seed=1)
  temp2 = temp()
  theta = get_laplacian(adj)
  print(theta.shape)
  X_tilde = random(k, n, density=0.15, random_state=1, data_rvs=temp2.rvs)
  C = random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs)
  X_t_0,C_0 = experiment(c_param[0],c_param[1],c_param[2],c_param[3],C,X_tilde,theta,X,p,n,k)
  L=theta
  C_t_0 = C_0.T
  C_0_new=torch.zeros(C_0.shape)
  for i in range(C_0.shape[0]):
      C_0_new[i][torch.argmax(C_0[i])] = 1

  Lc = C_0_new.T@L@C_0_new
  # Wc = (-1*Lc)*(1-torch.eye(Lc.shape[0]).cuda())
  Wc = (-1*Lc)*(1-torch.eye(Lc.shape[0]))
  # print(Wc.shape)
  # print(C_0_new.shape)

  Wc[Wc<0.1] = 0
  Wc = Wc.cpu().detach().numpy()
  Wc = sparse.csr_matrix(Wc)
  Wc = Wc.tocoo()
  row = torch.from_numpy(Wc.row).to(torch.long)
  col = torch.from_numpy(Wc.col).to(torch.long)
  edge_index_coarsen2 = torch.stack([row, col], dim=0)
  edge_weight = torch.from_numpy(Wc.data)
  Y = labels
  Y = one_hot(Y,NO_OF_CLASSES)

  P = torch.linalg.pinv(C_0_new)
  labels_coarse = torch.argmax(torch.sparse.mm(torch.Tensor(P).double() , Y.double()).double() , 1)

  Wc = Wc.toarray()
  try:
    X = torch.tensor(features.todense())
  except:
    X = torch.tensor(features)
  X1=X
  X1=torch.tensor(X1)
  Xt = P@X1
  x=sample(range(0, int(k)), k)
  return torch.Tensor(Xt), edge_index_coarsen2, labels_coarse


def preprocess(X, edge_index, y):
  adj = to_dense_adj(edge_index)
  adj = adj[0]
  labels = y
  labels = labels.numpy()

  # X = x
  X = X.to_dense()
  N = X.shape[0]
  features = X.numpy()
  NO_OF_CLASSES =  len(set(np.array(y)))

  print(X.shape, adj.shape)

  nn = int(1*N)
  X = X[:nn,:]
  adj = adj[:nn,:nn]
  labels = labels[:nn]
  return X,adj,labels,features,NO_OF_CLASSES

def coarse_graph_classification(X,adj, cr_ratio,c_param):
  features=X.numpy()
  p = X.shape[0]
  k = int(p*cr_ratio)   ###Corasening ratio
  n = X.shape[1]
  temp = CustomDistribution(seed=1)
  temp2 = temp()
  theta = get_laplacian(adj)
  print(theta.shape)
  X_tilde = random(k, n, density=0.15, random_state=1, data_rvs=temp2.rvs)
  C = random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs)
  X_t_0,C_0 = experiment(c_param[0],c_param[1],c_param[2],c_param[3],C,X_tilde,theta,X,p,n,k)
  L=theta
  C_t_0 = C_0.T
  C_0_new=torch.zeros(C_0.shape)
  for i in range(C_0.shape[0]):
      C_0_new[i][torch.argmax(C_0[i])] = 1

  Lc = C_0_new.T@L@C_0_new
  # Wc = (-1*Lc)*(1-torch.eye(Lc.shape[0]).cuda())
  Wc = (-1*Lc)*(1-torch.eye(Lc.shape[0]))
  # print(Wc.shape)
  # print(C_0_new.shape)

  Wc[Wc<0.1] = 0
  Wc = Wc.cpu().detach().numpy()
  Wc = sparse.csr_matrix(Wc)
  Wc = Wc.tocoo()
  row = torch.from_numpy(Wc.row).to(torch.long)
  col = torch.from_numpy(Wc.col).to(torch.long)
  edge_index_coarsen2 = torch.stack([row, col], dim=0)
  edge_weight = torch.from_numpy(Wc.data)
  
  P = torch.linalg.pinv(C_0_new)

  Wc = Wc.toarray()
  try:
    X = torch.tensor(features.todense())
  except:
    X = torch.tensor(features)
  X1=X
  X1=torch.tensor(X1)
  Xt = P@X1
  x=sample(range(0, int(k)), k)
  return torch.Tensor(Xt), edge_index_coarsen2
