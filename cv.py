import torch
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import data
import network

class CV():
    def __init__(
        self,
        path_train='data/train/', 
        path_val='data/validation/',
        epochs=15,
        n_splits=5,
        batch_size=64,
        targetsize=64,
        lr=0.01,
        weight_decay=0.1,
        momentum=0.9, 
        shuffle_dataset=True, 
        model_name='net0.pt',
        ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.targetsize = targetsize
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle_dataset = shuffle_dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.model_name = model_name
        self.path_train = path_train
        self.path_val = path_val
        self.n_splits = n_splits

    def get_dataset(self):
        # initialize transformations
        no_t = transforms.Compose([
            transforms.Resize((self.targetsize, self.targetsize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        t = transforms.Compose([
            transforms.Resize((self.targetsize, self.targetsize)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # initialize data
        dataset = data.RoadSignDataset(self.path_train, self.path_val, t, no_t)
        trainset, valset, num_of_classes = dataset.__getimages__()
        return trainset, valset, num_of_classes

    def weight_reset(self, m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    def train(self, net, optimizer, criterion, train_loader, device):
        running_loss = 0.0
        net.train()
        for idx, (inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.to(device).to(torch.float)
            labels = labels.to(device).to(torch.long)
            # clear the parameter gradient
            optimizer.zero_grad()
            # forward propagation
            outputs = net(inputs)
            #
            #score, prediction = torch.max(outputs, 1)
            # calculate the loss
            loss = criterion(outputs, labels)
            # back propagation
            loss.backward()
            # update model weights
            optimizer.step()
            # track loss
            running_loss += loss.item()
        return running_loss
    
    def validation(self, net, criterion, val_loader, device):
        running_loss = 0.0
        y_true = []
        y_pred = []
        net.eval()
        with torch.no_grad():
            for (inputs, labels) in val_loader:
                inputs = inputs.to(device).to(torch.float)
                labels = labels.to(device).to(torch.long)
                # forward propagation
                outputs = net(inputs)
                #
                #score, prediction = torch.max(outputs, 1)
                #y_pred += prediction.tolist()
                #y_true += labels.tolist()
                # calcuate & track loss
                loss = criterion(outputs, labels)
                running_loss += loss.item()
        return running_loss
        #return running_loss, y_pred, y_true

    def see_loss(self, train_losses, val_losses):
        plt.plot(train_losses, color='orange', linewidth=2, label='Training Loss')
        plt.plot(val_losses, color='red', linewidth=2, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="best")
        plt.grid()
        plt.show()

    def train_model(self, dataset_, num_of_classes):
        #
        kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle_dataset, random_state=42)
        train_losses = []
        val_losses = []
        #
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset_)):
            print('Fold {}'.format(fold + 1))

            train_subsampler = SubsetRandomSampler(train_ids)
            val_subsampler = SubsetRandomSampler(test_ids)

            train_loader = DataLoader(dataset_, batch_size=self.batch_size, sampler=train_subsampler)
            val_loader = DataLoader(dataset_, batch_size=self.batch_size, sampler=val_subsampler)

            net = network.Net(num_of_classes).to(self.device)
            net.apply(self.weight_reset)
        
            optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
            criterion = nn.CrossEntropyLoss()

            train_loss = []
            val_loss = []

            for epoch in range(self.epochs):
                # train
                loss = self.train(net, optimizer, criterion, train_loader, self.device)
                train_loss.append(loss)
                # validation
                loss = self.validation(net, criterion, val_loader, self.device)
                val_loss.append(loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        torch.save(net.state_dict(), self.model_name)

        train_losses = np.array(train_losses)
        val_losses = np.array(val_losses)
        train_losses_avg = np.mean(train_losses, axis=0)
        val_losses_avg = np.mean(val_losses, axis=0)
        self.see_loss(train_losses_avg, val_losses_avg, self.epochs)

    def run(self):
        # get data
        trainset, valset, num_of_classes = self.get_dataset()
        dataset_ = ConcatDataset([trainset, valset])
        # train/validate model
        self.train_model(dataset_, num_of_classes)

if __name__=="__main__":
    cv = CV()
    cv.run()