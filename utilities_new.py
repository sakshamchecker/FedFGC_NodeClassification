import torch
def average_a_list(li):
    return sum(li)/len(li)

def train_a_epoch(model, train_loader, optimizer, criterion,device, dp, priv_budget):
    model.train()
    model.to(device)
    for data in train_loader:  # Iterate in batches over the training dataset.
        #to device
        data.to(device)
        # scheduler.step()
        
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        if dp:
            delta=0.2
            # sigma=delta/(2*priv_budget)
            sigma=priv_budget
            clip_value=1.1
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.tensor(torch.randn_like(param.grad) * sigma)
                        param.grad += noise   
        #do optimizer and scheduler step
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        

def test(model,loader, batch_size):
    model.eval()
    criterion=torch.nn.CrossEntropyLoss()
    correct = 0 
    loss=0
    losses=[]
    corrects=[]
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        # print(data.x.shape)
        loss += criterion(out, data.y).item()
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        losses.append(criterion(out, data.y).item())
        corrects.append(int((pred == data.y).sum()))
    # print(total_graphs)
    # return average_a_list(corrects), average_a_list(losses)
    # if isinstance(loader.dataset, list):
    #     return loss/(len(loader.dataset)*batch_size), correct/(len(loader.dataset)*batch_size)
    return loss/len(loader.dataset), correct/len(loader.dataset)
    

def train(model,train_loader, test_loader,epochs, lr, device, dp, priv_budget, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    # scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch in range(1, epochs+1):
        train_a_epoch(model, train_loader, optimizer, criterion, device, dp, priv_budget)
        train_loss, train_acc = test(model, train_loader, batch_size)
        test_loss, test_acc = test(model, test_loader, batch_size)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    return model


def train(model, train_data, epochs, lr, device, dp, priv_budget):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_data.to(device)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)
        loss = criterion(out, train_data.y.type(torch.LongTensor).to(device))
        loss.backward()
        if dp:
            delta=0.2
            # sigma=delta/(2*priv_budget)
            sigma=priv_budget
            clip_value=1.1
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.tensor(torch.randn_like(param.grad) * sigma)
                        param.grad += noise 
        optimizer.step()
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        loss,acc=test(model,train_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

    return model

def test(model, test_data):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_data.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    out = model(test_data.x, test_data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred == test_data.y).sum()
    acc = int(correct) / int(test_data.y.shape[0])
    loss = criterion(out, test_data.y.type(torch.LongTensor).to(device))
    return loss.item(), acc