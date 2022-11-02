import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, classes):
        super().__init__() # just run the init of parent class (nn.Module)
        self.specrogram_width = 64
        self.specrogram_length = 2586
        # how many genres
        self.classes = classes
        
        # making the 3 2D convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 convolutional features, 5x5 kernel / window size
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels
        self.conv3 = nn.Conv2d(64, 128, 5)

        # changing convolutional layers to fully connected/linear layers (fc1 and fc2) by flattening
        # input spectrogram size, 1 picture at a time
        x = torch.randn(self.specrogram_width, self.specrogram_length)
        x = torch.randn(self.specrogram_width, self.specrogram_length).view(-1, 1, self.specrogram_width, self.specrogram_length) 
        self._to_linear = None
        self.convs(x)        
        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, self.classes) # 512 in, # of classes out

    def convs(self, x):
        # max pooling over 2x2, pooling is reducing the image size
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        # print(x[0].shape)

        # setting the shape of the data
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)

class Model():
    def __init__(self, batch_size, epochs, learning_rate, validation_percent, training_data):
        """Constuctor method for TrainModel class can tweak epoch batch_size"""

        #add training data path
        self.training_data = np.load(f"{training_data}", allow_pickle=True)

        # batch size is how many spectrographs it uses per pass
        self.batch_size = batch_size
        
        # epochs are how many times it runs through all of the data
        # more epochs will make our model fit the training data more
        self.epochs = epochs
        
        # learning rate
        self.learning_rate = learning_rate
        
        # validation sample size
        self.validation_percent = validation_percent

        self.specrogram_width = 64
        self.specrogram_length = 2586

    def train_model(self, classes):
        # if your computer is set up to train on the GPU it will use that, otherwise the CPU
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # 0 is the first GPU, can add more
            print("Running on the GPU")
        else:
            device = torch.device("cpu")
            print("Running on the CPU")

        # send to neural network to GPU
        net = Net(classes).to(device)

        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_function = nn.MSELoss()

        # seprating training and validation data
        # shape the data (inputting our data size in pixels) - pytorch expects (-1, IMG_SIZE, IMG_SIZE) format
        X = torch.Tensor([i[0] for i in self.training_data]).view(-1, self.specrogram_width, self.specrogram_length)
        # X = X/255.0
        y = torch.Tensor([i[1] for i in self.training_data])

        # seperate out testing and validation data
        val_size = int(len(X)*self.validation_percent)

        # group data into train and test sets
        train_X = X[:-val_size]
        train_y = y[:-val_size]

        test_X = X[-val_size:]
        test_y = y[-val_size:]
        #print(len(train_X), len(test_X), len(train_y), len(test_y))


        for epoch in range(self.epochs):
            # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            for i in tqdm(range(0, len(train_X), self.batch_size)): 
                #print(f"{i}:{i+self.batch_size}")
                batch_X = train_X[i:i+self.batch_size].view(-1, 1, self.specrogram_width, self.specrogram_length)
                batch_y = train_y[i:i+self.batch_size]
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                net.zero_grad()
                outputs = net(batch_X) # pass in the reshaped batch
                loss = loss_function(outputs, batch_y) # calc and grab the loss value
                loss.backward()    # apply this loss backwards thru the network's parameters
                optimizer.step()    # attempt to optimize weights to account for loss/gradients

            print(f"Epoch: {epoch}. Loss: {loss}")


            # testing
            correct = 0
            total = 0
            for i in tqdm(range(0, len(test_X), self.batch_size)):
                batch_X = test_X[i:i+self.batch_size].view(-1, 1, self.specrogram_width, self.specrogram_length).to(device)
                batch_y = test_y[i:i+self.batch_size].to(device)
                batch_out = net(batch_X)

                out_maxes = [torch.argmax(i) for i in batch_out]
                target_maxes = [torch.argmax(i) for i in batch_y]
                for i,j in zip(out_maxes, target_maxes):
                    if i == j:
                        correct += 1
                    total += 1
            print("Accuracy: ", round(correct/total, 3))