import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt 
import numpy as np
import data
import network
import metrics

class Model():
    def __init__(
        self,
        save_model,
        path_train='data/train/', 
        path_val='data/validation/',
        epochs=100,
        batch_size=64,
        targetsize=64,
        lr=0.01,
        weight_decay=0.1, #0.01
        momentum=0.9, 
        shuffle_dataset=True
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
        self.path_train = path_train
        self.path_val = path_val
        self.save_model = save_model

    def get_dataset(self):
        # initialize transformations
        t0 = transforms.Compose([
            transforms.Resize((self.targetsize, self.targetsize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        t1 =  transforms.Compose([
            transforms.Resize((self.targetsize, self.targetsize)),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomVerticalFlip(p=0.7),
            #transforms.GaussianBlur((3, 3), sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        transform = [t0, t1]
        # get data
        dataset = data.RoadSignDataset(self.path_train, self.path_val, transform)
        trainset, valset, num_of_classes, trainSample_dist, smoteSample_dist = dataset.get_images()
        # initialize data
        trainset_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
        valset_loader = DataLoader(valset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
        return trainset_loader, valset_loader, num_of_classes, trainSample_dist, smoteSample_dist

    def see_dataset(self, trainloader, classes):
        # get training images
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        # get/prep images
        img = make_grid(images)
        img = img / 2.0 + 0.5 # kinda-unnormalize
        # show images
        plt.imshow( (img.permute(1, 2, 0).numpy()*255).astype(np.uint8) )
        print( ', '.join( '%5s' % classes[labels[j]] for j in range(self.batch_size) ) )
        plt.show()

    # Tad inefficient
    def see_dataset_dist(self, trainSample_dist, smoteSample_dist, classes):
        ind = np.arange(start=0, stop=len(classes), step=1)
        width = 0.35
        for idx in ind:
            plt.bar(ind[idx], trainSample_dist[idx][0], width, color='r')
            plt.bar(ind[idx], trainSample_dist[idx][1], width, bottom=trainSample_dist[idx][0], color='g')
            plt.bar(ind[idx], trainSample_dist[idx][2], width, bottom=trainSample_dist[idx][0]+trainSample_dist[idx][1], color='b')
            try:
                plt.bar(ind[idx], smoteSample_dist[idx][0]+smoteSample_dist[idx][1]+smoteSample_dist[idx][2], width, bottom=trainSample_dist[idx][0]+trainSample_dist[idx][1]+trainSample_dist[idx][2], color='c')
            except IndexError:
                continue
        plt.xticks(range(len(ind)), classes, rotation=0)
        plt.ylabel('No. of Samples')
        plt.title('Image Distribution per Class')
        plt.grid(axis='y')
        plt.show()

    def see_weights(self, net, epoch):
        # get the kernels
        kernel = net.conv0.weight.cpu().detach().clone()
        # normalize to (0,1) range so that matplotlib can plot them
        kernel = kernel - kernel.min()
        kernel = kernel / kernel.max()
        kernel_image = make_grid(kernel, nrow = 32)
        # change ordering since matplotlib requires images to be (H, W, C)
        plt.imshow(kernel_image.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig('result\weights1\{}.jpg'.format(epoch), bbox_inches='tight', pad_inches=0.05, dpi=200)
        
    def train_model(self, net, dataset_loader, valset_loader, classes, optimizer, criterion):
        running_losses = []
        running_losses_val = []
        net.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            running_loss_val = 0.0
            for idx, (inputs, labels) in enumerate(dataset_loader, 0):
                inputs  = inputs.to(self.device).to(torch.float)
                labels = labels.to(self.device).to(torch.long)
                # clear the parameter gradient
                optimizer.zero_grad()
                # forward propagation
                output = net(inputs)
                # calculate the loss
                loss = criterion(output, labels)
                # back propagation
                loss.backward()
                # update model weights
                optimizer.step()
                # track loss
                running_loss += loss.item()
            running_losses.append(running_loss)

            running_loss_val = self.validate_model(net, valset_loader, criterion)
            running_losses_val.append(running_loss_val)

            print( 'Epoch: {}, Loss: {}'.format(epoch, np.round(running_loss, 2)) )
        return running_losses, running_losses_val
    
    def validate_model(self, net, valset_loader, criterion):
        running_loss = 0.0
        net.eval()
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(valset_loader, 0):
                inputs  = inputs.to(self.device).to(torch.float)
                labels = labels.to(self.device).to(torch.long)
                # forward propagation
                output = net(inputs)
                # calculate the loss
                loss = criterion(output, labels)
                running_loss += loss.item()
        return running_loss

    def see_loss(self, running_losses, running_losses_val):
        plt.plot(running_losses, color='orange', linewidth=2, label='Training Loss')
        plt.plot(running_losses_val, color='red', linewidth=2, label='Validation Loss' )
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss \n (Loss: {}, Epochs: {}, Lr: {}, Decay: {})'.format( round(running_losses[-1], 2), self.epochs, self.lr, self.weight_decay) )
        plt.legend(loc="best")
        plt.grid()
        plt.show()

    def run(self):
        # get data
        dataset_loader, valset_loader, num_of_classes, trainSample_dist, smoteSample_dist = self.get_dataset()
        classes = ['Speed limit (100kmh)', 'Speed limit (120kmh)', 'Speed limit (30kmh)', 'Speed limit (50kmh)', 'Speed limit (60kmh)', 'Speed limit (70kmh)', 'Speed limit (80kmh)']
        # initialize: net, hyperparameters
        net = network.Net(num_of_classes).to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        # see data
        self.see_dataset(dataset_loader, classes)
        self.see_dataset_dist(trainSample_dist, smoteSample_dist, classes)
        # train model
        running_losses, running_losses_val = self.train_model(net, dataset_loader, valset_loader, classes, optimizer, criterion)
        torch.save(net.state_dict(), self.save_model)
        self.see_loss(running_losses, running_losses_val)

if __name__=="__main__":
    model = Model(lr=0.0003, weight_decay=0.05, epochs=250, save_model='net1.pt')
    model.run()