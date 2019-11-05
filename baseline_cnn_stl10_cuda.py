from __future__ import division
import torch
import torch.utils.data as tud 
import torch.nn.functional as F
from torch import nn, optim
import torchvision.transforms as transforms
import numpy as np



class TwoLayerConvNet(nn.Module):
    """
    A 2-layer convolutional NN with dropout and batch-normalization 
    Dimension progression: 
        (if raw resolution = 128). 128*128*3 -> 128*128*10 -> 64*64*10 -> 64*64*20 -> 16*16*20 -> 64 -> 10 
    """
    def __init__(self, image_reso, filter_size, dropout_rate):
        super(TwoLayerConvNet, self).__init__()
        
        self.best_dev_accuracy = 0

        assert filter_size % 2 == 1, "filter_size = %r but it has to be an odd number" % filter_size
        
        # 3 input channels, 10 output channels, 5 x 5 filter size
        self.conv1_drop = nn.Dropout2d(dropout_rate)
        self.conv1 = nn.Conv2d(3, 10, kernel_size=filter_size, stride = 1, padding = (filter_size - 1)//2)  
        self.bn1 = nn.BatchNorm2d(num_features=10)
        
        # reduce spatial dimension by 2 times
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 0)   
        
        # 10 input channels, 20 output channels, 5 x 5 filter size
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=filter_size, stride = 1, padding = (filter_size - 1)//2)
        self.bn2 = nn.BatchNorm2d(num_features=20)
        
        # reduce spatial dimension by 4 times 
        self.avgpool = nn.AvgPool2d(kernel_size=4)  
        
        self.fc1 = nn.Linear(20 * (image_reso//8) * (image_reso//8), 64)
        self.fc2 = nn.Linear(64, 10)        
        
    def forward(self, x):
        
        
        # 3 x 96 x 96 -> 10 x 96 x 96 
        x = self.conv1_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        
        # 10 x 96 x 96 -> 10 x 48 x 48
        x = self.pool1(x)
        
        
        # 10 x 48 x 48 -> 20 x 48 x 48
        x = self.conv2_drop(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)     
        
        
        # 20 x 48 x 48 -> 20 x 12 x 12
        x = self.avgpool(x)
               
        # resize the 2d representation to a vector
        x = x.view(-1, 20 * 12 * 12)
    
        # 1 x (20 * 12 * 12) -> 1 x 64 
        x = F.relu(self.fc1(x))

        # 1 x 64 -> 1 x 10 
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class BaselineConvNet(object):
    """
    Packaged version of the baseline CNN model, including train and test function 
    Parameters:
    filter_size: filter size for ConvNet 
    dropout_rate: drop out rate for Conv layer 
    image_reso: 64, 128, or 256. The size of the input image 
    lr: learning rate for the optimizerfloat, learning rate (default: 0.001)
    batch_size: int, batch size (default: 128)
    cuda: bool, whether to use GPU if available (default: True)
    """
    def __init__(self, image_reso = 96, path = "baseline.pth", filter_size = 5, dropout_rate = .2, 
                 lr=1.0e-3, batch_size=10, cuda = True):
        
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        
        self.model = TwoLayerConvNet(image_reso, filter_size, dropout_rate)
        self.model.to(self.device)
        self.path = path
        self.image_reso = image_reso
        self.filter_size = filter_size
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.trainset_loader = None
        self.testset_loader = None

        self.initialize()

    def train(self, epoch, log_interval=100):
        self.model.train() 
        iteration = 0
        best_dev_accuracy = 0
        for ep in range(epoch):
            for batch_idx, (data, target) in enumerate(self.trainset_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()
                if iteration % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        ep, batch_idx * len(data), len(self.trainset_loader.dataset),
                        100. * batch_idx / len(self.trainset_loader), loss.item()))
                iteration += 1
            dev_accuracy = self.dev()
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                self.model.best_dev_accuracy = best_dev_accuracy
                torch.save(self.model, self.path)
    
    # dev set evaluation
    def dev(self):
        self.model.eval() 
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.testset_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.testset_loader.dataset)
        print('\nDev set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.testset_loader.dataset),
            100. * correct / len(self.testset_loader.dataset)))
        return correct / len(self.testset_loader.dataset)

    # test set evaluation
    def test(self,testset_loader, path, return_confusion_matrix = False):

        if not torch.cuda.is_available():
            self.model = torch.load(path, map_location='cpu')
        else:
            self.model = torch.load(path)
        self.model.eval() 
        correct = 0
        with torch.no_grad():
            confusion_matrix = torch.zeros(10, 10)
            for data, target in testset_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                
                for t, p in zip(target.view(-1), pred.view(-1)): #make confusion matrix
                    confusion_matrix[t.long(), p.long()] += 1
                    
                correct += pred.eq(target.view_as(pred)).sum().item()

        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(testset_loader.dataset),
            100. * correct / len(testset_loader.dataset)))
        
        if return_confusion_matrix:
            return (confusion_matrix,confusion_matrix.diag()/confusion_matrix.sum(1))
        

    def fit(self, train_loader, test_loader):
        
        self.trainset_loader = train_loader
        self.testset_loader = test_loader
        
        return 
    
    def initialize(self):
        """
        Model Initialization
        """
        nn.init.xavier_uniform_(self.model.conv1.weight)
        nn.init.xavier_uniform_(self.model.conv2.weight)
        nn.init.xavier_uniform_(self.model.fc1.weight)
        nn.init.xavier_uniform_(self.model.fc2.weight)
        
        return