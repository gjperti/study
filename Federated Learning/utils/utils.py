import torch
from torch.utils.data import DataLoader, random_split

def copy_global(server, clients):
    for client in clients:
        client.load_state_dict(server.state_dict())

def get_params(model):
    return torch.cat([p.view(-1,1) for p in model.parameters()]).detach()

def get_grads(model):
    return torch.cat([p.grad.view(-1,1) for p in model.parameters()]).detach()

def get_grads_2(model):
    return torch.cat([p.grad.view(-1,1) for p in model.parameters()])

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = correct / total
    return accuracy

def split_dataset(dataset, n_clients):
    """
    Splits the dataset into n_clients subsets.
    """
    lengths = len(dataset) // n_clients
    splits = [lengths] * n_clients
    train_loaders = [DataLoader(dataset, batch_size=64, shuffle=True) for dataset in random_split(dataset, splits)]
    return train_loaders

# class FedNetwork():
#     def __init__(self, n_clients, model):
#         self.clients = [model() for _ in range(n_clients)]
#         self.server = model()
#         self.n_params = len(self.get_params(self.server))
#         self.n_clients = n_clients

#     def copy_global(self):
#         for client in self.clients:
#             client.load_state_dict(self.server.state_dict())
    
#     def get_params(self, model):
#         return torch.cat([p.view(-1,1) for p in model.parameters()]).detach()

#     def get_grads(self, model):
#         return torch.cat([p.grad.view(-1,1) for p in model.parameters()]).detach()
    
#     def add_client(self, model):
#         self.clients.append(model())
#         self.n_clients += 1
    
#     def train_clients(self, train_loaders, optimizers, loss_func):
            
#         grads = torch.zeros(self.n_params, self.n_clients)

#         # Optimizing clients for 1 batch
#         for k, client in enumerate(self.clients):

#             client.train()

#             # Get a single batch of data
#             data,target = next(iter(train_loaders[k]))

#             # Batch is fed to the client
#             output = client(data)

#             # loss is calculated
#             loss = loss_func(output, target)

#             # loss gradient to each param is calculated
#             optimizers[k].zero_grad()
#             loss.backward()

#             # gradients are collected
#             grads[:,k] = get_grads(client).detach().view(-1)
            
#         return grads
    
#     def update_client(self, grads, lr=0.1):
#         j = 0
#         for param in self.server.parameters():
#             param.data -= lr*grads[j:j + param.numel(),:].mean(dim=1).view(param.shape)
#             j = j + param.numel()

class Client():

    def __init__(self, model, dataloader, id):
        self.model = model
        self.dataloader = dataloader
        self.id = id
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.n_params = len(self.get_params())

    def train(self):
        self.model.train()
        data, labels = next(iter(self.dataloader))
        
        outputs = self.model(data)
        loss = self.loss_func(outputs, labels)

        self.optimizer.zero_grad()
        loss.backward()

    def get_params(self):
        return torch.cat([p.view(-1,1) for p in self.model.parameters()]).detach()
    
    def get_grads(self, grad=False):
        if grad:
            return torch.cat([p.grad.view(-1,1) for p in self.model.parameters()])
        else:   
            return torch.cat([p.grad.view(-1,1) for p in self.model.parameters()]).detach()
    
    def set_params(self, server):
        self.model.load_state_dict(server.model.state_dict())

    def __call__(self,x):
        return self.model(x)
    
    def __repr__ (self):
        return f"Client {self.id} with model\n\n {self.model}"
    
class Server():

    def __init__(self, model):
        self.model = model
        self.clients = []
        self.n_params = len(self.get_params())

    def get_params(self):
        return torch.cat([p.view(-1,1) for p in self.model.parameters()]).detach()
    
    def update_params_from_grad(self, grads, lr):
        j = 0
        for param in self.model.parameters():
            param.data -= lr*grads[j:j + param.numel(),:].mean(dim=1).view(param.shape)
            j += param.numel()

    def __call__(self, x):
        return self.model(x)
    
class FedNetwork():
    def __init__(self, clients, server):
        self.clients = clients
        self.server = server
        self.n_params = server.n_params
        self.n_clients = len(clients)
        self.client_ids = [client.id for client in clients]
    
    def add_client(self, client):
        self.clients.append(client)
        self.n_clients += 1
        self.client_ids.append(client.id)

    def initialize_clients(self):
        for client in self.clients:
            client.set_params(self.server)
    
    def train_clients(self):
            
        grads = torch.zeros(self.n_params, self.n_clients)
        params = torch.zeros(self.n_params, self.n_clients)

        # Optimizing clients for 1 batch
        for k, client in enumerate(self.clients):
            client.train()
            grads[:,k] = (client).get_grads().view(-1)
            params[:,k] = client.get_params().view(-1)
            
        return params, grads
    
    def update_client(self, grads, lr=0.1):
        self.server.update_params_from_grad(grads, lr)

    def drop_client(self, client_id):
        if client_id in self.client_ids:
            index = self.client_ids.index(client_id)    
            del self.clients[index]
            del self.client_ids[index]
            self.n_clients -= 1
        else:
            raise IndexError("Client ID not found in the network.")

    def __repr__(self):
        return f"Federated Network with {self.n_clients} clients and server model\n\n {self.server.model}"