#general VAE Structure from https://qiita.com/koshian2/items/b6402d3b4494dcd6d6a1
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
import numpy as np
from IPython.core.debugger import set_trace



class EncoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Encoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        super().__init__()
        self.bottle = EncoderModule(color_channels, 32, stride=1, kernel=1, pad=0)
        self.m1 = EncoderModule(32, 64, stride=1, kernel=3, pad=1)
        self.m2 = EncoderModule(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.m3 = EncoderModule(128, 256, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x):
        out = self.m3(self.m2(self.m1(self.bottle(x))))
        return out.view(-1, self.n_neurons_in_middle_layer)
        
class DecoderModule(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.convt = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.bn(self.convt(x)))

    
class Decoder(nn.Module):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        self.decoder_input_size = decoder_input_size
        super().__init__()
        self.m1 = DecoderModule(256, 128 , stride=1)
        self.m2 = DecoderModule(128, 64, stride=pooling_kernels[1])
        self.m3 = DecoderModule(64, 32, stride=pooling_kernels[0])
        self.bottle = DecoderModule(32, color_channels, stride=1, activation="sigmoid")

    def forward(self, x):
        out = x.view(-1, 256, self.decoder_input_size, self.decoder_input_size)
        out = self.m3(self.m2(self.m1(out)))
        return self.bottle(out)

    
class M2_base(nn.Module):
    def __init__(self, latent_features, classifier, num_labels = 10):
        """Structure of the M2 model
        Parameters
        ----------
        number of latent features,
        classifier for the model
        """
        super().__init__()
        
        self.n_latent_features = latent_features
        self.best_dev_accuracy = 0

        pooling_kernel = [4, 4]
        encoder_output_size = 6
        color_channels = 3
        

        # neurons int middle layer
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size

        # Encoder
        self.encoder = Encoder(color_channels, pooling_kernel, n_neurons_middle_layer)
        # Middle
        self.fc1 = nn.Linear(n_neurons_middle_layer + num_labels, self.n_latent_features )
        self.fc2 = nn.Linear(n_neurons_middle_layer + num_labels, self.n_latent_features)
        self.fc3 = nn.Linear(self.n_latent_features + num_labels, n_neurons_middle_layer)
        # Decoder
        self.decoder = Decoder(color_channels, pooling_kernel, encoder_output_size)
        self.classifier = classifier

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def _bottleneck(self, h, y):
        mu, logvar = self.fc1(torch.cat([h,y.float()],dim = 1)), self.fc2(torch.cat([h,y.float()],dim = 1))
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x, y):
        # Encoder
        h = self.encoder(x)
        # Bottle-neck
        z, mu, logvar = self._bottleneck(h,y)
        # decoder
        z = self.fc3(torch.cat([z,y],dim = 1))
        d = self.decoder(z)
        return d, mu, logvar


    
def log_standard_categorical(p):
    """Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    Parameters
    ----------
    p: one-hot categorical distribution
    """
    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)

    return cross_entropy

# -L(x,y), elbo for labeled data
def _L(x,y,recon_x,mu,logvar):
    n, d = mu.shape
    loglik = -F.binary_cross_entropy(recon_x, x, reduction='sum')/n
    KLD = -0.5*(d + (logvar-logvar.exp()).sum()/n - mu.pow(2).sum()/n)
    loglik_y = -log_standard_categorical(y).sum()/n
    
    return loglik + loglik_y - KLD

    
# -U(x), elbo for unlabeled data
def _U(x,log_prob,recon_x,mu,logvar):
    n, d = mu.shape
    
    prob = torch.exp(log_prob)
    
    #Entropy of q(y|x)
    H = -torch.mul(prob,log_prob).sum(1).mean()
    
    # -L(x,y)
    loglik = -F.binary_cross_entropy(recon_x, x, reduction='none').sum(1).sum(1).sum(1) #n*1
    KLD = -0.5*(d + (logvar-logvar.exp()) - mu.pow(2)).sum(1) #n*1
    
    
    if (x.is_cuda):
        y = torch.cuda.FloatTensor((1,0,0,0,0,0,0,0,0,0)).reshape(1,-1)
    else:
        y = torch.FloatTensor((1,0,0,0,0,0,0,0,0,0)).reshape(1,-1)

    loglik_y = -log_standard_categorical(y)  #constant, value same for all y since we have a uniform prior
    
    _Lxy = loglik + loglik_y - KLD #n*1
    
    # sum q(y|x) * -L(x,y) over y
    q_Lxy = torch.sum(prob * _Lxy.view(-1,1))/n
    
    return q_Lxy + H


class M2(object):
    """M2 model 
    Parameters
    ----------
    number of latent features,
    classifier for the model,
    lr: learning rate,
    cuda: whether to use GPU if available (default: True),
    path: path to store model
    """
    def __init__(self, latent_features, classifier, lr=1.0e-3, cuda=True, path="m2.pth"):
        self.model = M2_base(latent_features,classifier)
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.lr = lr
        self.path = path
        self.initialize
    
        
    def fit(self, train_loader,test_loader, epochs,alpha,labeled_data_len):
        """Fit M2
        Parameters
        ----------
        train_loader, 
        test_loader,
        epochs,
        alpha: weight for classifier loss,
        labeled_data_len: length of labeled data in dev and training set
        """
        best_dev_accuracy = 0
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(1, epochs + 1):
            train_loss,accuracy = self._train(train_loader,test_loader,optimizer,epoch,alpha,labeled_data_len)
            dev_accuracy = self._evaluate(test_loader)
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                self.model.best_dev_accuracy = best_dev_accuracy
                torch.save(self.model, self.path)
            print('Epoch: %d, train loss: %.4f, training accuracy %.4f, dev set accuracy %.4f' % (
                epoch, train_loss, accuracy,dev_accuracy))
        return


    def _train(self,train_loader,test_loader,optimizer,epoch,alpha,labeled_data_len):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        train_loss = 0
        correct = 0
        for batch_idx, data in enumerate(train_loader):
            data, labels = data
            data = data.to(self.device)
            #labels = labels.to(self.device)
            
            #make one-hot encoding for labels
            y_onehot = torch.FloatTensor(len(labels),11)
            y_onehot.zero_()
            y_onehot.scatter_(1,labels.view(-1,1),1)
            y_onehot = y_onehot.narrow(1,0,10)
            y_onehot = y_onehot.to(self.device)
            
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = self.model(data,y_onehot)
            
            #split data into unlabeled data and labeled data for computing loss
            index_n = torch.nonzero(labels == 10).reshape(-1)
            index_n = index_n.to(self.device)
            index = torch.nonzero(labels != 10).reshape(-1)
            index = index.to(self.device)
            unlabeled_data = data.index_select(0,index_n)
            labeled_data = data.index_select(0,index)
            recon_unlabeled = recon_batch.index_select(0,index_n)
            recon_labeled = recon_batch.index_select(0,index)
            mu_unlabeled = mu.index_select(0,index_n)
            mu_labeled = mu.index_select(0,index)
            logvar_unlabeled = logvar.index_select(0,index_n)
            logvar_labeled = logvar.index_select(0,index)
            
            y_onehot = y_onehot.index_select(0,index)
            
            # if no labeled data
            if ( sum(labels != 10) == 0):
                Lxy = 0
            else:
                # -Elbo for labeled data (L(X,y))
                Lxy = - _L(labeled_data,y_onehot,recon_labeled,mu_labeled,logvar_labeled)
            
            
            # get q(y|x) from classifier
            
            log_prob = self.model.classifier(unlabeled_data)
            
            # -Elbo for unlabeled data (U(x))
            Ux = - _U(unlabeled_data,log_prob,recon_unlabeled,mu_unlabeled,logvar_unlabeled)
            
            
            # Add auxiliary classification loss q(y|x)
            
            # if no labeled data
            if ( sum(labels != 10) == 0):
                classication_loss = 0
            else:
                log_prob = self.model.classifier(labeled_data)
                # negative cross entropy
                classication_loss = -torch.sum(y_onehot * log_prob, dim=1).mean()
            
            
            # The overall loss
            loss = Lxy + alpha * classication_loss + Ux
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # if there is labeled data in this batch
            if ( sum(labels != 10) > 0):
                target = labels.index_select(0,index.to("cpu"))
                target = target.to(self.device)
                pred = log_prob.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
           
        
            if batch_idx%50 == 0:
                print(batch_idx, len(train_loader), f"Loss: {train_loss/(batch_idx+1):f}")
                
            if batch_idx == len(train_loader)-1:
                if ( sum(labels != 10) > 0):
                        save_image(recon_labeled, f"reconstruction_epoch_{str(epoch)}.png", nrow=8)
                        save_image(labeled_data, f"truth_epoch_{str(epoch)}.png", nrow=8)
        
        return train_loss/(batch_idx+1), correct/(labeled_data_len-len(test_loader.dataset))
            

    def _evaluate(self,test_loader):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                
                #all data are labeled
                data, labels = data
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                log_prob = self.model.classifier(data)
            
            
                pred = log_prob.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()
        

        return correct/len(test_loader.dataset)
    
       
    # test set evaluation
    def test(self,test_loader, path, return_confusion_matrix = False):
        
        if not torch.cuda.is_available():
            self.model = torch.load(path, map_location='cpu')
        else:
            self.model = torch.load(path)
        self.model.eval()
        correct = 0
        with torch.no_grad():
            confusion_matrix = torch.zeros(10, 10)
            for batch_idx, data in enumerate(test_loader):
                
                #all data are labeled
                data, labels = data
                data = data.to(self.device)
                labels = labels.to(self.device)
                log_prob = self.model.classifier(data)
                
                pred = log_prob.max(1, keepdim=True)[1] # get the index of the max log-probability

                for t, p in zip(labels.view(-1), pred.view(-1)): #make confusion matrix
                    confusion_matrix[t.long(), p.long()] += 1
            
                correct += pred.eq(labels.view_as(pred)).sum().item()
        
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        if return_confusion_matrix:
            return (confusion_matrix,confusion_matrix.diag()/confusion_matrix.sum(1))

    
    
    def sampling(self,y_onehot):
        """Sample one image from model
        Parameters
        ----------
        onehot vector of labels,
        path of model
        """
        # assume latent features space ~ N(0, 1)
        try:
            self.model = torch.load(self.path)
        except Exception as err:
            print("Error loading '%s'\n[ERROR]: %s\nUsing initial model!" % (self.path, err))
            
        z = torch.randn(1,self.model.n_latent_features).to(self.device)
        z = self.model.fc3(torch.cat([z,y_onehot],dim = 1))
        # decode
        return self.model.decoder(z)
    
    
    def initialize(self):
        """
        Model Initialization
        """
        def _init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.model.apply(_init_weights)
        return
       
    
class Classifier(nn.Module):
    """
    A 2-layer convolutional NN with dropout and batch-normalization 
    Dimension progression: 
        (if raw resolution = 96). 96*96*3 -> 96*96*10 -> 48*48*10 -> 48*48*20 -> 12*12*20 -> 64 -> 10 
    """
    def __init__(self, image_reso, filter_size, dropout_rate):
        super(Classifier, self).__init__()
        
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

    

    
    