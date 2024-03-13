import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool
import copy
from collections import OrderedDict
# def train(net, trainloader,epochs, device):
#     criterion=nn.CrossEntropyLoss()
#     optimizer=optim.Adam(net.parameters(), lr=0.01)
#     best_loss=0
#     best_weights=[]
#     for epoch in range(1, epochs):
#         # train_per_epoch(trainloader,net, optimizer, criterion, device)
#         net=train_per_epoch(training_graphs=trainloader,model=net, optimizer=optimizer, criterion=criterion, device=device)
#         # train_acc = test(trainloader,net, device)
#         train_loss, train_acc = test(model=net, testloader=trainloader, device=device)
#         print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f} Train Loss: {train_loss:.4f}')
#         if train_acc>=best_loss:
#             best_loss=train_acc
#             best_weights=get_parameters(net)
#     net=set_parameters(net,best_weights)
#     return best_weights, best_loss



def train_ep(args, model, device, train_graphs, optimizer, epoch, dp, clip_value, sigma):
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    if args:
        batch_size = args.batch_size
    else:
        batch_size = 32
    total_iters = 100
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()  
            if dp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # Add noise to gradients
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.tensor(torch.randn_like(param.grad) * sigma)
                        param.grad += noise       
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        # pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    # print("loss training: %f" % (average_loss))
    
    return average_loss, model
def train(args, model, device, train_graphs, epochs, test_graphs, optimizer=None, dp=False, priv_budget=2):
    # if optimizer is None:
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # if args.data=='MUTAG':
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) #for MUTAG 20ep
    # else:
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) #for PTC 
    # minloss is infinity at first
    min_loss=float('inf') 
    # min_loss=0
    epsilon = priv_budget  # Privacy budget
    delta = 0.2  # Target delta

    # Define the noise scale
    sigma=delta/(2*epsilon)

    # Set gradient clipping threshold
    clip_value = 1.1
    best_weights=None
    for epoch in range(1, epochs + 1):
        # scheduler.step()
        print('---------Curr Accuracy------', min_loss)
        avg_loss, model= train_ep(args, model, device, train_graphs, optimizer, epoch, dp, clip_value, sigma,)
        model.eval()
        test_loss, test_acc=test(args, model, device, test_graphs)
        #check if the model is best, if yes, save the weights
        # if test_acc<=min_loss:
        #     min_loss=test_acc
        #     model.train()
        #     best_weights=copy.deepcopy(model)
        print(f'Epoch: {epoch:03d}, Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    # set_parameters(model, best_weights)
    best_weights=copy.deepcopy(model)
    return best_weights, min_loss
    # return avg_loss
    
# def train_per_epoch(training_graphs,model, optimizer, criterion, device):
    

#     model.train()
#     model.to(device)
#     for data in training_graphs:  # Iterate in batches over the training dataset.
#          out = model(x=data.x.to(device),edge_index= data.edge_index.to(device))
#         #  pred = out.argmax(dim=1)  # Perform a single forward pass.
#          out=global_mean_pool(out,data.batch)
#          loss = criterion(out, data.y)  # Compute the loss.
#          loss.backward()  # Derive gradients.
#          optimizer.step()  # Update parameters based on gradients.
#          optimizer.zero_grad()  # Clear gradients.
#     return model
def accuracy(y_true, y_pred):
    accuracy = np.mean(y_pred == y_true)
    return accuracy
# def test(model, testloader, device):
     
    model.eval()
    model.to(device)
    correct = 0
    accuracies=[]
    total_graphs=0
    running_loss=0.0
    criterion=nn.CrossEntropyLoss()
    for data in testloader:  # Iterate in batches over the training/test dataset.
        # total_graphs+=data.num_graphs
        # print(total_graphs)
        out = model(x=data.x.to(device), edge_index=data.edge_index.to(device)) 
        out=global_mean_pool(out,data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        #calculate accuracy with pred and data.y
        total_graphs+=len(pred)
        correct += int((pred == data.y).sum())
        # accuracies.append(accuracy(pred, data.y))
        loss = criterion(out, data.y)  # Compute the loss.
        running_loss+=loss.item()
    # print(total_graphs)
    return running_loss / len(testloader.dataset), correct/total_graphs

def tranc_floating(x):
    return float("{:.5f}".format(x))
def get_parameters(model):
    return [val.cpu().numpy() for name, val in model.state_dict().items() if 'num_batches_tracked' not in name]
def set_parameters(model,parameters):
        keys = [k for k in model.state_dict().keys() if 'num_batches_tracked' not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, test_graphs):
    model.eval()
    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))
    #calculate crossentropy loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, labels)
    loss_test=loss.item()
    # print("loss : %f accuracy: %f" % (loss_test, acc_test))
    # print("accuracy : %f test: %f" % (, acc_test))

    return loss_test, acc_test


def plot(experiment_path, c_methods, args):
    #open history and plot loss and accuracy 
    with open(f"{experiment_path}/history_{c_methods[0]}.pkl", "rb") as f:
        history = torch.load(f)
    with open(f"{experiment_path}/history_{c_methods[1]}.pkl", "rb") as f:
        history1 = torch.load(f)
    #plot loss on y axis and round on x axis
    #plt.plot([i[0] for i in history.losses_distributed], [i[1] for i in history.losses_distributed])
    plt.clf()
    plt.ylim(-2,2)
    # plt.ylim(min([i[0] for i in history.losses_distributed+history1.losses_distributed]), max([i[0] for i in history.losses_distributed+history1.losses_distributed]))
    plt.plot([i[0] for i in history.losses_distributed], label=" without coarsen")
    plt.plot([i[0] for i in history1.losses_distributed], label=" with coarsen")
    plt.grid(True)
    #add title
    plt.title(f"Loss vs Round aggregated")
    #add x and y labels
    plt.xlabel("Round")
    plt.ylabel("Loss")
    #add legend
    plt.legend()
    plt.savefig(f"{experiment_path}/loss.png")
    plt.clf()
    plt.ylim(0,1)
    plt.plot([i[0] for i in history.metrics_distributed["accuracy"]], [i[1] for i in history.metrics_distributed["accuracy"]], label=" without coarsen")
    plt.plot([i[0] for i in history1.metrics_distributed["accuracy"]], [i[1] for i in history1.metrics_distributed["accuracy"]], label=" with coarsen")
    #add x and y labels
    plt.grid(True)
    plt.title(f"Accuracy vs Round aggregated")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{experiment_path}/accuracy.png")

    #open every folder in client wise, and use coarse_false.txt and coarse_true.txt to plot loss and accuracy
    for j in range(args.ncl):
        #open coarse_false.txt and coarse_true.txt, they are in format, each line is a round, 0th index is round number, 1st is loss, 2nd is accuracy 
        with open(f"{experiment_path}/clientwise/{j}/coarse_False.txt", "r") as f:
            coarse_false=f.readlines()
        with open(f"{experiment_path}/clientwise/{j}/coarse_True.txt", "r") as f:
            coarse_true=f.readlines()
        #plot loss on y axis and round on x axis
        plt.clf()
        plt.ylim(-2,2)
        plt.plot([i.split(",")[0] for i in coarse_false], [i.split(",")[1] for i in coarse_false], label=" without coarsen")
        plt.plot([i.split(",")[0] for i in coarse_true], [i.split(",")[1] for i in coarse_true], label=" with coarsen")
        #add x and y labels
        plt.grid(True)
        plt.title(f"Loss vs Round client {j}")
        #set y lim between min and max of loss
        plt.xlabel("Round")
        plt.ylabel("Loss")
        #add legend
        plt.legend()
        plt.savefig(f"{experiment_path}/clientwise/{j}/loss.png")
        plt.clf()
        plt.ylim(0,1)
        plt.plot([i.split(",")[0] for i in coarse_false], [i.split(",")[2] for i in coarse_false], label=" without coarsen")
        plt.plot([i.split(",")[0] for i in coarse_true], [i.split(",")[2] for i in coarse_true], label=" with coarsen")
        #add x and y labels
        plt.grid(True)
        plt.title(f"Accuracy vs Round client {j}")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{experiment_path}/clientwise/{j}/accuracy.png")
