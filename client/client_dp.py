import argparse
import os

import tensorflow as tf

import flwr as fl


from collections import OrderedDict
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from utilities import tranc_floating
from utilities_new import train, test
from privacy.dp import dp as DiffP
import os
import torch
import copy
import time
# Define Flower client
from attack.membership_infer.attack_utils import attack_test, attack_train, data_creator

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloaders,trainloaders_cr, valloader, epochs, path,state, device, args, dp, priv_budget, attack_net):
        self.cid = int(cid)
        # self.model = net(params[0], params[1], params[2])
        self.model = net
        # GCN(hidden_channels=32, in_channels=num_node_features, out_channels=num_classes, num_layers=3)
        # self.model=net(hidden_channels=params[1], in_channels=params[0], out_channels=params[2], num_layers=3)
        self.trainloader_non = trainloaders
        self.trainloader_cr=trainloaders_cr
        if state:
            self.trainloader=self.trainloader_cr
        else:
            self.trainloader=self.trainloader_non
        self.valloader = valloader
        self.epochs = epochs
        self.device = device
        self.path = path
        self.state = state
        self.args = args
        self.dp=dp
        self.priv_budget=priv_budget
        self.attack_net=attack_net
    def get_parameters(self, config):
        return [val.cpu().numpy() for name, val in self.model.state_dict().items() if 'num_batches_tracked' not in name]

    def set_parameters(self, parameters):
        keys = [k for k in self.model.state_dict().keys() if 'num_batches_tracked' not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        #start time logger
        start = time.time()

        print(f"Training client {self.cid}")
        self.set_parameters(parameters)
        # params,avg_loss=train(args=self.args, model=self.model, device=self.device, train_graphs=self.trainloader[self.cid], epochs=self.epochs, test_graphs=self.valloader, dp=self.dp, priv_budget=self.priv_budget)
        # train(model=self.model, train_loader=self.trainloader, test_loader=self.valloader, epochs=self.epochs, lr=self.args.lr, device=self.device, dp=self.dp, priv_budget=self.priv_budget, batch_size=self.args.batch_size)
        train(model=self.model, train_data=self.trainloader, epochs=self.epochs, lr=self.args.lr, device=self.device, dp=self.dp, priv_budget=self.priv_budget)
        # loss, accuracy = test(args=self.args, model=self.model, device=self.device, test_graphs=self.trainloader[self.cid])
        loss, accuracy=test(self.model, self.trainloader)
        end=time.time()
        print(f"Time taken for client {self.cid} is {end-start}")
        try:
            data=pd.read_csv(f"{self.path}/results_train.csv")
            data.drop(["Unnamed: 0"], axis=1, inplace=True)
        except:
            data=pd.DataFrame(columns=["Method","Coarsen","Priv","Data","Round", "Client Number", "Loss","Accuracy", "Time"])
        data=pd.concat([data, pd.Series(['FL', self.state, self.dp, "Train",config['server_round']-1, self.cid, tranc_floating(loss), tranc_floating(accuracy), end-start], index=data.columns).to_frame().T], ignore_index=True)
        data.to_csv(f"{self.path}/results_train.csv")
        
        # loss, accuracy = test(self.model, self.valloader, self.device)
        # loss,accuracy = test(args=self.args, model=self.model, device=self.device, test_graphs=self.valloader)
        
        return self.get_parameters({}), self.trainloader.y.shape[0], {}
    def evaluate(self, parameters, config):
        print(f"Evaluating client {self.cid}")
        self.set_parameters(parameters)
        # loss,accuracy = test(self.model, self.valloader, self.device)
        # loss, accuracy = test(args=self.args, model=self.model, device=self.device, test_graphs=self.valloader)
        loss, accuracy = test(self.model, self.valloader)

        try:
            os.mkdir(f"{self.path}/clientwise/{self.cid}")
            print('-------FILE CREATED---------')
        except:
            print("")
        #append in file
        with open(f"{self.path}/clientwise/{self.cid}/coarse_{self.state}_{self.dp}.csv", "a") as f:
            f.write(f"{config['server_round']},{tranc_floating(loss)},{tranc_floating(accuracy)}\n")
        try:
            data=pd.read_csv(f"{self.path}/results.csv")
            data.drop(["Unnamed: 0"], axis=1, inplace=True)
        except:
            data=pd.DataFrame(columns=["Method","Coarsen","Priv","Data","Round", "Client Number", "Loss","Accuracy"])
        data=pd.concat([data, pd.Series(['FL', self.state, self.dp, "Test",config['server_round']-1, self.cid, tranc_floating(loss), tranc_floating(accuracy)], index=data.columns).to_frame().T], ignore_index=True)
        data.to_csv(f"{self.path}/results.csv")
        print(f"----Attacking {self.cid}---")
        for original in [True, False]:
            if original:
                target_train=self.trainloader_non
            else:
                if self.state:
                    target_train=self.trainloader_cr
                else:
                    target_train=self.trainloader_non
            attack_train_loader, attack_test_loader=data_creator(target_model=self.model, shadow_model=None, target_train=target_train, target_test=self.valloader, shadow_train=None, shadow_test=None, device=self.device)
            criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(attack_net.parameters(), lr=0.01,weight_decay=0.0001)
            test_loss, test_accuracy, final_auroc, final_precision, final_recall, final_f_score=attack_test(model=self.attack_net, testloader=attack_test_loader, device=self.device, trainTest=False, criterion=criterion)
            try:
                data = pd.read_csv(f"{self.path}/clientwise/{self.cid}/coarse_{self.state}_{self.dp}_attack.csv")
                data.drop(["Unnamed: 0"], axis=1, inplace=True)
            except:
                data = pd.DataFrame(columns=["Round","Coarsen","Privacy","Original","Loss","Accuracy","AUROC","Precision","Recall", "Final_f_score"])
            data=pd.concat([data, pd.Series([config['server_round'], self.state,self.dp, original, tranc_floating(test_loss), tranc_floating(test_accuracy), tranc_floating(final_auroc), tranc_floating(final_precision), tranc_floating(final_recall), tranc_floating(final_f_score)], index=data.columns).to_frame().T], ignore_index=True)
            data.to_csv(f"{self.path}/clientwise/{self.cid}/coarse_{self.state}_{self.dp}_attack.csv")
        return loss, len(self.valloader), {"accuracy": accuracy}