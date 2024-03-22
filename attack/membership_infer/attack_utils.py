import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, roc_curve, roc_auc_score, f1_score
import pandas as pd
def attack_train(model, trainloader, testloader,device, criterion, optimizer, epochs, steps=0):
    # train ntwk

    # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    final_train_loss = 0
    train_losses, train_accuracy_li = [], []
    posteriors = []
    model.to(device)
    for e in range(epochs):
        running_loss = 0
        train_accuracy = 0

        # This is features, labels cos we dont care about nodeID during training! only during test
        for features, labels in trainloader:
            model.train()
            features, labels = features.to(device), labels.to(device)

            # print("post shape", features.shape)
            # print("labels",labels)
            optimizer.zero_grad()
            # print("features", features.shape)

            # features = features.unsqueeze(1) #unsqueeze
            # flatten features
            features = features.view(features.shape[0], -1)

            logps = model(features)  # log probabilities
            # print("labelsssss", labels.shape)
            loss = criterion(logps, labels)

            # Actual probabilities
            ps = logps  # torch.exp(logps) #Only use this if the loss is nlloss
            # print("ppppp",ps)

            top_p, top_class = ps.topk(1,
                                        dim=1)  # top_p gives the probabilities while top_class gives the predicted classes
            # print(top_p)
            equals = top_class == labels.view(
                *top_class.shape)  # making the shape of the label and top class the same
            train_accuracy += torch.mean(equals.type(torch.FloatTensor))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_losses.append(running_loss/ len(trainloader))
        train_accuracy_li.append(train_accuracy/ len(trainloader))
        print(f"Epoch {e} Accuracy {train_accuracy/ len(trainloader)} ; Loss {running_loss/ len(trainloader)}")

    return sum(train_accuracy_li)/len(train_accuracy_li), sum(train_losses)/len(train_losses)

def attack_test(model, testloader,criterion,device, singleClass=False, trainTest=False):
    test_loss = 0
    test_accuracy = 0
    auroc = 0
    precision = 0
    recall = 0
    f_score = 0

    posteriors = []
    all_nodeIDs = []
    true_predicted_nodeIDs_and_class = {}
    false_predicted_nodeIDs_and_class = {}

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        # Doing validation

        # set model to evaluation mode
        model.eval()

        if trainTest:
            for features, labels in testloader:
                features, labels = features.to(device), labels.to(device)
                # features = features.unsqueeze(1)  # unsqueeze
                features = features.view(features.shape[0], -1)
                logps = model(features)
                test_loss += criterion(logps, labels)

                # Actual probabilities
                ps = logps  # torch.exp(logps)
                posteriors.append(ps)

                # if singleclass=false
                if not singleClass:
                    y_true = labels.cpu().unsqueeze(-1)
                    # print("y_true", y_true)
                    y_pred = ps.argmax(dim=-1, keepdim=True)
                    # print("y_pred", y_pred)

                    # uncomment this to show AUROC    device='cpu'
                    auroc += roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
                    # print("auroc", auroc)

                    precision += precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                    recall += recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                    f_score += f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                top_p, top_class = ps.topk(1,
                                            dim=1)  # top_p gives the probabilities while top_class gives the predicted classes
                # print(top_p)
                equals = top_class == labels.view(
                    *top_class.shape)  # making the shape of the label and top class the same
                test_accuracy += torch.mean(equals.type(torch.FloatTensor))

        else:
            # print("len(testloader.dataset)", len(testloader.dataset))
            for features, labels, nodeIDs in testloader:
                # print("nodeIDs", nodeIDs)
                features, labels = features.to(device), labels.to(device)
                # features = features.unsqueeze(1)  # unsqueeze
                features = features.view(features.shape[0], -1)
                logps = model(features)
                test_loss += criterion(logps, labels)

                # Actual probabilities
                ps = logps  # torch.exp(logps)
                posteriors.append(ps)

                # print("ps", ps)
                # print("nodeIDs", nodeIDs)

                all_nodeIDs.append(nodeIDs)

                # if singleclass=false
                if not singleClass:
                    y_true = labels.cpu().unsqueeze(-1)
                    # print("y_true", y_true)
                    y_pred = ps.argmax(dim=-1, keepdim=True)
                    # print("y_pred", y_pred)

                    # uncomment this to show AUROC
                    auroc += roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
                    # print("auroc", auroc)

                    precision += precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                    recall += recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                    f_score += f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                top_p, top_class = ps.topk(1,
                                            dim=1)  # top_p gives the probabilities while top_class gives the predicted classes
                # print("top_p", top_p)
                # print("top_class", top_class)

                equals = top_class == labels.view(
                    *top_class.shape)  # making the shape of the label and top class the same

                # print("equals", len(equals))
                for i in range(len(equals)):
                    if equals[i]:  # if element is true {meaning both member n non-member}, get the nodeID
                        # print("baba")
                        # print("true pred nodeIDs", nodeIDs[i])
                        true_predicted_nodeIDs_and_class[nodeIDs[i].item()] = top_class[i].item()
                        # print("len(true_predicted_nodeIDs_and_class)", len(true_predicted_nodeIDs_and_class),"nodeID--",nodeIDs[i].item(), "class--",  top_class[i].item())
                    else:
                        false_predicted_nodeIDs_and_class[nodeIDs[i].item()] = top_class[i].item()
                        # print("len(false_predicted_nodeIDs_and_class)", len(false_predicted_nodeIDs_and_class))

                test_accuracy += torch.mean(equals.type(torch.FloatTensor))

    test_accuracy = test_accuracy / len(testloader)
    test_loss = test_loss / len(testloader)
    final_auroc = auroc / len(testloader)
    final_precision = precision / len(testloader)
    final_recall = recall / len(testloader)
    final_f_score = f_score / len(testloader)
    return test_loss, test_accuracy, final_auroc, final_precision, final_recall, final_f_score

def sub_data_creator(model, train_loader, test_loader, isTarget,device):
    model.to(device)
    train_loader.to(device)
    test_loader.to(device)
    pred = model(train_loader.x, train_loader.edge_index)
    pred_Intrain = pred.max(1)[1]
    # Actual probabilities
    # pred_Intrain_ps = torch.exp(model(data_new.target_x,data_new.target_edge_index)[data_new.target_train_mask])
    intrain = torch.exp(pred).detach().cpu().numpy()
    pred = model(test_loader.x, test_loader.edge_index)
    outtrain=torch.exp(pred).detach().cpu().numpy()
    intrain=pd.DataFrame(intrain)
    outtrain=pd.DataFrame(outtrain)
    if isTarget:
        intrain['nodeID'] = range(0,intrain.shape[0])
        outtrain['nodeID'] = range(0,outtrain.shape[0])
        
    intrain['labels']=1
    outtrain['labels']=0
    
    data=pd.concat([intrain,outtrain])
    return data

def data_creator(target_model, shadow_model, target_train, target_test, shadow_train, shadow_test,device):
    attack_train_data= sub_data_creator(model=shadow_model, train_loader=shadow_train, test_loader=shadow_test, isTarget=False, device=device)
    attack_train_data_X=attack_train_data.drop(["labels"], axis=1)
    attack_train_data_y=attack_train_data['labels']
    attack_train_data_X, attack_train_data_y=attack_train_data_X.to_numpy(), attack_train_data_y.to_numpy()
    attack_train_data=torch.utils.data.TensorDataset(torch.from_numpy(attack_train_data_X).float(), torch.from_numpy(
        attack_train_data_y))
    attack_train_data_loader = torch.utils.data.DataLoader(attack_train_data, batch_size=32, shuffle=True)
    attack_test_data = sub_data_creator(model=target_model, train_loader=target_train, test_loader=target_test, isTarget=True, device=device )
    attack_test_data_X=attack_test_data.drop(["labels","nodeID"], axis=1)
    attack_test_data_y=attack_test_data['labels']
    attack_test_data_node=attack_test_data['nodeID']
    attack_test_data_X, attack_test_data_y, attack_test_data_node=attack_test_data_X.to_numpy(), attack_test_data_y.to_numpy(), attack_test_data_node.to_numpy()
    # attack_test_data_X, attack_test_data_y=attack_test_data_X.to_numpy(), attack_test_data_y.to_numpy()
    attack_test_data=torch.utils.data.TensorDataset(torch.from_numpy(attack_test_data_X).float(), torch.from_numpy(
        attack_test_data_y),torch.from_numpy(attack_test_data_node))
    attack_test_data_loader = torch.utils.data.DataLoader(attack_test_data, batch_size=32, shuffle=True)

    return attack_train_data_loader, attack_test_data_loader
# 