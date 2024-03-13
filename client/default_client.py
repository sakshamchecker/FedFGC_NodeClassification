import flwr as fl
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
import torch
import sys
import pandas as pd
sys.path.append('..')
from utilities import train, test, tranc_floating
import os
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloaders, valloader, epochs, path,state, device, args):
        self.cid = int(cid)
        # self.model = net(params[0], params[1], params[2])
        self.model = net
        # GCN(hidden_channels=32, in_channels=num_node_features, out_channels=num_classes, num_layers=3)
        # self.model=net(hidden_channels=params[1], in_channels=params[0], out_channels=params[2], num_layers=3)
        self.trainloader = trainloaders
        self.valloader = valloader[self.cid]
        self.epochs = epochs
        self.device = device
        self.path = path
        self.state = state
        self.args = args    
    # def get_parameters(self, config):
    #     self.model.eval()
    #     return [val.cpu().numpy() for _, val in self.model.state_dict().items()] 
    
    # def set_parameters(self, parameters):
    #     self.model.train()
    #     params_dict = zip(self.model.state_dict().keys(), parameters)
    #     state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    #     self.model.load_state_dict(state_dict, strict=True)
    def get_parameters(self, config):
        return [val.cpu().numpy() for name, val in self.model.state_dict().items() if 'num_batches_tracked' not in name]

    def set_parameters(self, parameters):
        keys = [k for k in self.model.state_dict().keys() if 'num_batches_tracked' not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"Training client {self.cid}")
        self.set_parameters(parameters)
        # train(self.model, self.trainloader[self.cid], self.epochs, self.device)
        avg_loss=train(args=self.args, model=self.model, device=self.device, train_graphs=self.trainloader[self.cid], epochs=self.epochs, test_graphs=self.valloader)
        # loss, accuracy = test(self.model, self.trainloader[self.cid], self.device)
        loss, accuracy = test(args=self.args, model=self.model, device=self.device, test_graphs=self.trainloader[self.cid])
        try:
            data=pd.read_csv(f"{self.path}/results_train.csv")
            data.drop(["Unnamed: 0"], axis=1, inplace=True)
        except:
            data = pd.DataFrame(columns=["Method","Coarsen","Data","Round", "Client Number", "Loss","Accuracy"])
        data=pd.concat([data, pd.Series(['FL', self.state, "Train",config['server_round']-1, self.cid, tranc_floating(loss), tranc_floating(accuracy)], index=data.columns).to_frame().T], ignore_index=True)
        data.to_csv(f"{self.path}/results_train.csv")
        # loss, accuracy = test(self.model, self.valloader, self.device)
        loss,accuracy = test(args=self.args, model=self.model, device=self.device, test_graphs=self.valloader)
        
        return self.get_parameters({}), len(self.trainloader[self.cid]), {}


    def evaluate(self, parameters, config):
        print(f"Evaluating client {self.cid}")
        self.set_parameters(parameters)
        # loss,accuracy = test(self.model, self.valloader, self.device)
        loss, accuracy = test(args=self.args, model=self.model, device=self.device, test_graphs=self.valloader)
        # accuracy = 0

        try:
            os.mkdir(f"{self.path}/clientwise/{self.cid}")
            print('-------FILE CREATED---------')
        except:
            print("")
        #append in file 
        with open(f"{self.path}/clientwise/{self.cid}/coarse_{self.state}.txt", "a") as f:
            f.write(f"{config['server_round']},{tranc_floating(loss)},{tranc_floating(accuracy)}\n")
        try:
            data=pd.read_csv(f"{self.path}/results_test.csv")
            data.drop(["Unnamed: 0"], axis=1, inplace=True)
        except:
            data = pd.DataFrame(columns=["Method","Coarsen","Data","Round", "Client Number", "Loss","Accuracy"])
        data=pd.concat([data, pd.Series(['FL', self.state, "Test",config['server_round']-1, self.cid, tranc_floating(loss), tranc_floating(accuracy)], index=data.columns).to_frame().T], ignore_index=True)
        data.to_csv(f"{self.path}/results_test.csv")
        
       
        
        # try:
        #     data=pd.read_csv(f"{self.path}/results_test.csv")
        #     data.drop(["Unnamed: 0"], axis=1, inplace=True)
        # except:
        #     data = pd.DataFrame(columns=["Method","Coarsen","Data","Round", "Client Number" "Loss","Accuracy"])
        # data=pd.concat([data, pd.Series(['FL', self.state, "Test",config['server_round'], self.cid, tranc_floating(loss), tranc_floating(accuracy)], index=data.columns).to_frame().T], ignore_index=True)
        # data.to_csv(f"{self.path}/results_test.csv")
        return loss, len(self.valloader), {"accuracy": accuracy}