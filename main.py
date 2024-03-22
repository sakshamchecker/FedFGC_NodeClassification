from client.client_dp import FlowerClient
from server.server import fit_config, evaluate_config, FedAvgWithAccuracyMetric, FedProxWithAccuracyMetric, FedOptAdamStrategy
import flwr as fl
import torch
import torchvision
from models.GCN import GCN
from models.GIN import GraphCNN
# from torch_geometric.nn import GCN,GIN
import os
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
from data.loader import split_graphs, load_graphs, load_data, separate_data, load_data_pre
from data.loader_new import load_base_data, load_central_data, load_clients_data
from utilities import train, test, tranc_floating, get_parameters, plot, set_parameters
from utilities_new import train, test
import matplotlib.pyplot as plt
from privacy.dp import dp as DiffP
from attack.membership_infer.attack_model import Net
from attack.membership_infer.attack_utils import attack_test, attack_train, data_creator
#epochs
# epochs = 20
import copy
torch.manual_seed(42)
np.random.seed(42)
import time
import warnings
warnings.filterwarnings('ignore')

#set random seeed

#load trainloaders, and valloader

def run(args):
    experiment_path = f"{args.output}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.ncl}_{args.rounds}_{args.tr_ratio}_{args.epochs}_{args.data}_{args.strat}_Coarsen_{args.coarsen}_{args.cr_ratio}_Privacy_{args.privacy}_PrivBudget_{args.priv_budget}"
    print(f"Experiment path: {experiment_path}")
    os.makedirs(experiment_path, exist_ok=True)
    if args.privacy=='All' or args.privacy=='all':
        p_methods=[False, True]
    elif args.privacy=='False' or args.privacy=='false':
        p_methods=[False]
    else:
        p_methods=[True]
    if args.coarsen=='All' or args.coarsen=='all':
        c_methods=[False, True]
    elif args.coarsen=='False' or args.coarsen=='false':
        c_methods=[False]
    else:
        c_methods=[True]

    for i in c_methods:
        for j in p_methods:
            loss, accuracy = execute(args=args, coarsen=i, path=experiment_path, priv=j)
    # idxs=None
    # for i in c_methods:
    #     for j in p_methods:
    #         start_time = time.time()
    #         history,idxs=execute_FL(args, i, experiment_path,idxs, j)
    #         end_time=time.time()
    #         #save history to .pkl
    #         with open(f"{experiment_path}/history_{i}_{j}.pkl", "wb") as f:
    #             torch.save(history, f)
    #         #save history to txt
    #         with open(f"{experiment_path}/history_{i}_{j}.txt", "w") as f:
    #             f.write(str(history))
    #         try:
    #             data = pd.read_csv(f"{experiment_path}/results_test.csv")
    #             data.drop(["Unnamed: 0"], axis=1, inplace=True)
    #         except:
    #             data = pd.DataFrame(columns=["Method","Coarsen","Privacy", "Data","Round","Client Number", "Loss","Accuracy", "time"])
    #         curr=pd.DataFrame(columns=['Round Number', 'Loss', 'Accuracy'])
    #         #save history in a difference csv for each c_methods
    #         for k in range(len(history.metrics_distributed["accuracy"])):
    #             curr = pd.concat([curr, pd.Series([k, history.losses_distributed[k][-1], history.metrics_distributed["accuracy"][k][-1]], index=curr.columns).to_frame().T], ignore_index=True)
    #         curr.to_csv(f"{experiment_path}/history_{i}_{j}.csv")
    #         data = pd.concat([data, pd.Series(['FL_aggregated', i, j, 'Test', 'aggregated', '-' ,tranc_floating(history.losses_distributed[-1][-1]), tranc_floating(history.metrics_distributed["accuracy"][-1][-1]), end_time-start_time], index=data.columns).to_frame().T], ignore_index=True)
    #         data.to_csv(f"{experiment_path}/results_test.csv")
    # plot(experiment_path, c_methods, args)
def execute(args, coarsen, path, priv):
    train_data,train_data_cr, val_loader,shadow_train, shadow_test, num_node_features, num_classes=load_central_data(args.data, args.tr_ratio, cr=coarsen, cr_ratio=args.cr_ratio)
    if args.process == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    start=time.time()
    net = GCN(num_node_features, num_classes).to(device)
    end=time.time()
    if coarsen:
        train_loader=train_data_cr
    else:
        train_loader=train_data
    print('-- Training Target Model --')
    mod=train(model=net, train_data=train_loader, epochs=args.epochs, lr=args.lr, dp=priv, priv_budget=args.priv_budget, device=device)
    net=copy.deepcopy(mod)
    loss, accuracy=test(model=net, test_data=train_loader)
    try:
        data = pd.read_csv(f"{path}/results_train.csv")
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
    except:
        data = pd.DataFrame(columns=["Method","Coarsen","Privacy", "Data","Round","Client Number", "Loss","Accuracy", 'Time'])
    data = pd.concat([data, pd.Series(['AllData', coarsen,priv,"Train", 0, 0, tranc_floating(loss), tranc_floating(accuracy), end-start], index=data.columns).to_frame().T], ignore_index=True)
    data.to_csv(f"{path}/results_train.csv")
    loss, accuracy=test(model=net, test_data=val_loader)
    try:
        data = pd.read_csv(f"{path}/results_test.csv")
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
    except:
        data = pd.DataFrame(columns=["Method","Coarsen","Privacy","Data","Round","Client Number", "Loss","Accuracy", "Time"])
    data = pd.concat([data, pd.Series(['AllData', coarsen,priv,"Test", 0, 0, tranc_floating(loss), tranc_floating(accuracy), 0], index=data.columns).to_frame().T], ignore_index=True)
    data.to_csv(f"{path}/results_test.csv") 
    print(f'Accuracy: {accuracy:.4f} Loss: {loss:.4f}')
    print('-- Training Shadow Model --')
    shadow_net = GCN(num_node_features, num_classes).to(device)
    mod=train(model=shadow_net, train_data=shadow_train, epochs=args.epochs, lr=args.lr, dp=False, priv_budget=args.priv_budget, device=device)
    shadow_net=copy.deepcopy(mod)
    print('-- Training Attack Model --')
    attack_train_loader, attack_test_loader=data_creator(target_model=net, shadow_model=shadow_net, target_train=train_data, target_test=val_loader, shadow_train=shadow_train, shadow_test=shadow_test, device=device)
    attack_net=Net(num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(attack_net.parameters(), lr=0.01,weight_decay=0.0001)
    attack_train(model=attack_net, trainloader=attack_train_loader, testloader=attack_test_loader, device=device, criterion=criterion, optimizer=optimizer, epochs=args.epochs, steps=0)
    # del net, train_loader, val_loader
    test_loss, test_accuracy, final_auroc, final_precision, final_recall, final_f_score=attack_test(model=attack_net, testloader=attack_test_loader, device=device, trainTest=False, criterion=criterion)
    try:
        data = pd.read_csv(f"{path}/attack_test.csv")
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
    except:
        data = pd.DataFrame(columns=["Method","Loss","Accuracy","AUDOC","Precision","Recall", "Final_f_score"])
    data=pd.concat([])
    return loss, accuracy


    
    
def execute_FL(args, coarsen, path, idxs=None, priv=False):
    # train_loaders, valloader, num_classes, idxs = load_clients_data(dataset=args.data, clients=args.ncl, alpha=args.alpha, coarsen=coarsen, idxs=idxs, cr_ratio=args.cr_ratio)    
    # if not idxs:
        # train_loaders, test_loaders,num_node_features, num_classes, idxs=load_clients_data(data_name=args.data, batch_size=args.batch_size, client_number=int(args.ncl), tr_ratio=args.tr_ratio, alpha=args.alpha, cr=coarsen, client_data=idxs, cr_ratio=args.cr_ratio)
    train_loaders, test_loaders, num_node_features, num_classes=load_clients_data(args.data, args.ncl, args.tr_ratio, cr=coarsen, cr_ratio=args.cr_ratio)
    # with open(f"{path}/idxs.txt", "w") as f:
    #     for i in range(len(idxs)):
    #         f.write(f"{i} {len(idxs[i])}\n")
    if args.process == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    print("-------------------Data Loaded")
    # net=GraphCNN(num_layers=5, num_mlp_layers=2, input_dim=train_loaders[0][0].node_features.shape[1], hidden_dim=64, output_dim=num_classes, final_dropout=0.5, learn_eps=False, graph_pooling_type="sum", neighbor_pooling_type="sum", device=device).to(device)
    # net=GCN(num_node_features=num_node_features, hidden_channels=32, num_classes=num_classes).to(device)
    net=GCN(num_node_features,num_classes).to(device)
    try:
        os.mkdir(f"{path}/clientwise")
    except:
        print("")
    def client_fn(cid):
        return FlowerClient(cid, net, train_loaders[int(cid)], test_loaders, args.epochs, path=path, state=coarsen, device=device, args=args, dp=priv, priv_budget=args.priv_budget) 
    ray_args = {'num_cpus':4, 'num_gpus':1}
    client_resources = {"num_cpus": 4, "num_gpus": 1}
    if args.strat=="FedAvg":
        st=FedAvgWithAccuracyMetric(
            min_available_clients=int(args.ncl),
            # initial_parameters=get_parameters(net),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
        )
    elif args.strat=='FedProx':
        st=FedProxWithAccuracyMetric(
            min_available_clients=int(args.ncl),
            # initial_parameters=get_parameters(net),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            proximal_mu=0.1,
        )
    elif args.strat=='FedOptAdam':
        st=FedOptAdamStrategy(
            min_available_clients=int(args.ncl),
            # initial_parameters=get_parameters(net),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
        )
    history=fl.simulation.start_simulation(
        client_fn=client_fn,
        strategy=st,
        num_clients=int(args.ncl),

        config=fl.server.ServerConfig(num_rounds=int(args.rounds)),
        ray_init_args=ray_args,
        client_resources=client_resources,
    )
    
    del net, train_loaders, test_loaders
    return history, idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flower CIFAR10')
    parser.add_argument('--ncl', default=4, type=int, help='number of clients')
    parser.add_argument('--rounds', default=10, type=int, help='number of rounds')
    parser.add_argument('--output', default='output', type=str, help='output folder')
    parser.add_argument('--tr_ratio', default=0.8, type=float, help='train ratio')
    parser.add_argument('--process', default='cuda', type=str, help='cpu or gpu')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--data', default='XIN', type=str, help='dataset')
    parser.add_argument('--alpha', default=10, type=float, help='alpha')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--strat', default='FedAvg', type=str, help='strategy')
    parser.add_argument('--privacy', default="False", type=str, help='privacy')
    parser.add_argument('--coarsen', default="False", type=str, help='coarsen')
    parser.add_argument('--cr_ratio', default=0.5, type=float, help='coarsen ratio')
    parser.add_argument('--priv_budget', default=0.15, type=float, help='privacy budget')
    parser.add_argument('--lr' , default=0.1, type=float, help='learning rate')
    
    args = parser.parse_args()
    run(args)

#python3 main.py --ncl 2 --rounds 10 --output output --tr_ratio 0.8 --process cpu --epochs 50 --data Tox21_AhR_training