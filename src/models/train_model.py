
import numpy as np               # For using numpy arrays
from tqdm import tqdm            # Used for displaying progress bar during runtime
import torch                     # Main ML framework
import torch.optim as optim      # To use torch's optimizer
import torch.nn as nn            # To use torch's neural network class
import torch.nn.functional as F  # To use a variety built in functions such as activator
from torchvision import transforms, datasets
import random
import time
from src.data.PrepareInput import PrepareAudio
import json

"""
----Instructions----
The main class of concern is the Model class, of which the train_model is the only  
method that needs to be called since it makes calls to other methods appropriately.
Note, the Net class is not meant to be directly called. 

----Terminology----
-'Loss' refers to the degree of error (Context: you are getting only 50% match for a rock song, when you want it higher)
-An 'Optimizer' is what adjusts the weights for the purpose of lowering the loss after each run through the NN
-The 'Learning Rate' is the rate (size of the step) that the optimizer uses to make adjustments to model to minimize loss
-A single 'epoch' is 1 pass through dataset
-'Validation' refers to the process of evaluating how well a machine model was able to make predictions (Accuracy)

----Values we need to tweak to optimize model----
1. batch size: This is relative to size of whole dataset. For a dataset of 50, I have been using a value of 3-5
2. epoch count: This is a bit tricky. I thought that the more runs through the NN you do, the better the model becomes (I was wrong lol) Accuracy seems to peak after a certain point and then rapidly decreases. The peak epoch was variable for me. I think this is due to the randomization or shuffling of the dataset
3. learning rate: I stuck to 0.0001. I was getting some weird behavior when I attempted to use slightly larger values
4. validation percent: I think 0.1 (10%) is the sweet spot. 
5. Number of neurons in convolutional network: My system couldnt handle anything more than 3 lol but I think this could be a value worth playing around with as well.
6. Number of fully connected layers (maybe): I am unsure about this one. Leaving it at just 2 layers is probably best to decrease complexity, especially with a zillion other factors in play.

----Issues----
1. Couldnt chart output from training and testing
2. Couldnt construct neural network of convolutional neurons iteratively in the constructor method of the Net class. So each neuron had to be defined line-by-line
"""



class Net(nn.Module):
    """Representation of a Netural Network."""

    def __init__(self, classes, spec_width, spec_length):
        """Constructor class for Net class."""

        super().__init__()   # Specifies you want to access feature of the parent class.

        # Initialize variables
        self.classes = classes    # Number of music genres
        self.spec_width = spec_width
        self.spec_length = spec_length

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize Convolutional neural network
        self.conv1 = nn.Conv2d(1, 32, 5)   # Conv2d args: input, output (# of convolutional features), kernel size (makes x by x size window that rolls over the image to find features)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv_neurons = [self.conv1, self.conv2, self.conv3]    # Initialize a neuron array (Minimum 2)

        """ **My failed attempt a creating a convolutional NN through iteration (My ideas was that we can pass in an integer that represents
        neuron count and construct a working NN on the fly). This unfortunately throws an error so using the above code for now:

        # Initialize Convolutional neural network
        self.conv_neurons = []   # Initialize a neuron array
        in_val, out_val = 1, 32  # Initialize input and output
        kernel_size = 5          # Creates * by * size window that rolls over the image to find features
        for _ in range(3):
            neuron = nn.Conv2d(in_val, out_val, kernel_size)
            self.conv_neurons.append(neuron)
            in_val = out_val      # Update input val  (The output of previous neuron)
            out_val = out_val*2   # Update output val (Number of convolutional features)
        """

        # Find initial input into fully connected neural network by using Tensor of arbitrary values
        x = torch.randn(self.spec_width, self.spec_length)     # Initialize tensor of random values that is identical in shape to a spectrogram
        x = x.view(-1, 1, self.spec_width, self.spec_length)   # Flatten tensor

        # Call convs to determine initial input into fully connected neural network
        self._to_linear = None
        self.convs(x)    # Call conv() to set self._to_linear aka initial input

        # Initialize fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, self.classes)
        self.fc_layers = [self.fc1, self.fc2]  # Initialize a layer array (Minimum 2)

    def convs(self, x):
        """Runs data through each convolutional neuron consecutively. The output of one 
        will serve as the input for the next and so on. Also sets self._to_linear.
        *Meant to be called only by Net class (self)
        
        args: Tensor
        return: The result of running tensor through each consecutive neuron
        """ 

        # Run data through each layer in convolutional NN, accumulating (aka pooling) detected features
        for neuron in self.conv_neurons:
            x = F.max_pool2d(F.relu(neuron(x)), (2, 2))

        # Set class variable self._to_linear
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        
        return x

    def forward(self, x):
        """
        The function through which training/testing will occur
        *Called automatically every time the class instance is called. Not meant to be directly called. 

        args: x= Tensor
        return: probability distribution of run
        """

        x = self.convs(x)                 # Get output from running data through convolutional NN
        x = x.view(-1, self._to_linear)   # Flatten/reshape Tensor

        # Pass data through each (expect last) Fully Connected (FC) layer, which in turn is put through an activation function (relu aka Rectified Linear)
        for layer in range(len(self.fc_layers)-1):
            fcl = self.fc_layers[layer]   # Current Fully Connected Layer
            x = F.relu(fcl(x))

        x = self.fc_layers[-1](x)    # Run through last FC layer (the final layer does not need to be run through an activation function) 
        return F.softmax(x, dim=1)   # Return log of probability distrubution of # of values in dataset




class Model:
    """The representation of our music genre classifier machine model."""

    def __init__(self, batch_size, epochs, learning_rate, validation_percent, data_path, dict_path):
        """Constructor method for the Modle class.
        
        Args: *Detailed below.
        """

        # Values that will likely need tweaking to optimize model (Trial and Error)
        self.batch_size = batch_size    # Size of data passed through NN at a time
        self.epochs = epochs            # Number of runs through dataset
        self.learning_rate = learning_rate             # Rate (size of step) that the optimizer uses to make adjustments to mode to minimize loss
        self.validation_percent = validation_percent   # Value that will be modified to represent % of dataset that we will be training/testing
        self.spec_width = 64         # Spectrogram dimension
        self.spec_length = 2586      # Spectrogram dimension

        self.data_path = data_path              # Directory path to data (.npy)
        self.genre_dict = self.genre_file_to_dict(dict_path)    # Genre dictionary with genre names
        self.classes = len(self.genre_dict)     # Number of genres
        self.device = self.get_device()         # Initialize hardware to run model on (CPU or GPU)
        self.net = Net(self.classes, self.spec_width, self.spec_length).to(self.device)   # Initialize neural net instance
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)  # Initialize optimizer. Args(adjustable parameters, learning rate)
        self.loss_function = nn.MSELoss()   # Initialize loss function (Mean Squared Error is the one commonly used for one hot vectors)
        self.model_name = "Model_MGR_1"     # Model name for writing data to .log file

    def train_model(self):
        """Main driver function which allows for training and testing of a dataset.
        """
        
        # 1. Load Numpy dataset
        print("--> Loading numpy dataset...")
        data = self.get_data()

        # 2. Convert the data from numpy to a Tensor
        print("--> Converting dataset to tensors...")
        dataset, labelset = self.convert_numpy_to_tensor(data)

        # 3. Scale dataset
<<<<<<< HEAD
        print("--> Scaling dataset...")
        dataset = self.scale_set(dataset)
=======
        # dataset = self.scale_set(dataset)
>>>>>>> dd74b95e865d625b5e68d3e851fbd2f2d06a36a9

        # 4. Slice a portion of both train dataset and train labelset for training
        val_size = int(len(dataset)*self.validation_percent)    # Get int for slicing dataset
        train_x = dataset[:-val_size]
        train_y = labelset[:-val_size]

        # 5. Slice a portion of both test dataset and test labelset for testing/validating
        test_x = dataset[-val_size:]
        test_y = labelset[-val_size:]

        # 6. Train and Test [Custom implementation]
        print("--> Training dataset...")
        self.train_and_test(train_x, train_y, test_x, test_y) 
        return

        # 6 v2. Train and Test [Original]    *NONFUNCTIONAL This throws errors*
        self.train_and_test_v2(train_x, train_y, test_x, test_y)



    # Dataset management/manipulation methods
    def get_device(self):
        """Called by constructor once to set the device on which trainig/testing will occur.
        return: device instance to run model (i.e. CPU or cuda GPU)
        """

        """xxx"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # 0 is the first GPU, can add more
            print("Running on the GPU")
        else:
            device = torch.device("cpu")
            print("Running on the CPU")
        return device
        
    def get_data(self):
        """Loads the numpy array.
        return: Loaded dataset
        """
        return np.load(f"{self.data_path}", allow_pickle=True)

    def convert_numpy_to_tensor(self, data):
        """Converts the numpy array to a Tensor so that it can be utilzed by the model.
        args: data= loaded numpy dataset
        return: dataset tensor and label tensor   
        """
        # Create tensor from dataset
        data_tensor = torch.Tensor([i[0] for i in data])
        data_tensor = data_tensor.view(-1, self.spec_width, self.spec_length) # Reshape tensor

        # Create tensor of labels (aka one hot vectors)
        label_tensor = torch.Tensor([i[1] for i in data])

        return (data_tensor, label_tensor)

    def scale_set(self, set):
        """Scales the entire dataset so that all value are between 0 and 1.
        args: datasor (tensor)
        return: scaled tensor
        """
        # Scale all images so that all pixel values are between 0 and 1. This is done by dividing all values by the max val
        max_val = 0   # Find maximum value across all Tensors
        for genre in set:
            for song in genre: 
                for pixel in song:
                    if pixel > max_val:
                        max_val = pixel
        return set/max_val    # Divide all values across Tensors by the largest value and return


    # Training and testing (validation) methods
    def train_and_test(self, train_x, train_y, test_x, test_y):
        """Oversees the training and validation process.
        args: train_x= training dataset, train_y = training labelset
              test_x= validation dataset, test_y = validation labelset
        """

        for e in range(self.epochs):   # Make self.epoch number of runs through trainset/testset      
            matches = 0   # Tracks number of correct predictions
            total = 0     # Tracks total number of elements. Value will differ based on how the testset was split (regardless of batch size)

            # Train + test in batches
            for i in range(0, len(train_x), self.batch_size):
                # A. Use slicing to construct train batch
                train_batch_x = train_x[i:i+self.batch_size]       # Get dataset batch
                train_batch_x = train_batch_x.view(-1, 1, self.spec_width, self.spec_length)  # Reshape tensor
                train_batch_y = train_y[i:i+self.batch_size]       # Get label batch
                train_batch_x, train_batch_y = train_batch_x.to(self.device), train_batch_y.to(self.device)   # Load onto choosen device

                # B. Use slicing to construct test batch
                test_batch_x = test_x[i:i+self.batch_size]
                test_batch_x = test_batch_x.view(-1, 1, self.spec_width, self.spec_length)   # Get data in batch and load to GPU
                test_batch_y = test_y[i:i+self.batch_size].to(self.device)             # Get label pertaining to current batch and load to GPU
                test_batch_x, test_batch_y = test_batch_x.to(self.device), test_batch_y.to(self.device)   # Load onto choosen device

                # C. Train (Run trainset through NN)
                batch_loss = self.train_mod(train_batch_x, train_batch_y)

                # D. Test/validate (Run testset through NN)
                batch_matches, batch_total, l = self.test_mod(test_batch_x, test_batch_y)

                # E. Update counters
                matches += batch_matches
                total += batch_total

            # Print to visualize changes post epoch run
            print(f"Epoch[{e}/{self.epochs}: Loss= {round(float(batch_loss), 5)},Accuracy[{int(round(matches/total, 5)*100)}%]= {matches}/{total}")

    def train_mod(self, batch_x, batch_y):
        """Runs data through the neural network for the purpose of training.
        args: batch_x= slice of original dataset, bach_y = slice of corresponding labelset
        return: loss from training
        """

        # Zero the gradient. The gradient stores your losses, allowing your optimizer to look through it and
        # ... optimize weights accordingly. Since we initialized the optimizer with the net class parameters, 
        # ... we can just call zero_grad on the optimizer rather that the NN
        self.optimizer.zero_grad()   # Call before every time you pass the data through the Neural Network

        output = self.net(batch_x)  # Run batch through NN
        loss = self.loss_function(output, batch_y)   # Get loss from output
        loss.backward()         # Backwards propagate the loss
        self.optimizer.step()   # Perform the actual optimization
        
        return loss

    def test_mod(self, batch_x, batch_y):
        """Runs data through the neural network for the purpose of testing/validation.
        args: batch_x= slice of original dataset, bach_y = slice of corresponding labelset
        return: a tuple consisting of the accuracy, total values in batch, and loss that resulted from testing
        """

        with torch.no_grad():  # Run without gradient since we are not training
            batch_out = self.net(batch_x)   # Run through NN

        out_maxes = [torch.argmax(i) for i in batch_out]   # Get output from NN
        target_maxes = [torch.argmax(i) for i in batch_y]  # Variabilize correct matches

        # Check if any predictions were correct and record
        correct = 0 
        total = 0 
        for i,j in zip(out_maxes, target_maxes):
            if i == j:
                correct += 1
            total += 1
        
        # Compute the loss
        loss = self.loss_function(batch_out, batch_y)

        return (correct, total, loss)     # Return accuracy of determination(s)


    # Prototype/Unused method from original implmentation
    def train_and_test_v2(self, train_x, train_y, test_x,  test_y):
        """Alternative version of train_and_test.

        *NON-FUNCTIONAL
        """

        model_log = open('model.log', 'a')  # Initialize .log file to store stats onto
        create_file = True         # Set to False if you do not want to log data
        for epoch in range(self.epochs):

            # for i in tqdm(range(0, len(train_x), self.batch_size)):  # Uncomment to display progress bar during runtime
            for i in range(0, len(train_x), self.batch_size):          # Uncomment to run without dispaying progrress bar
                batch_x = train_x[i:i+self.batch_size].view(-1,1,self.spec_width,self.spec_length)
                batch_y = train_y[i:i+self.batch_size]

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                acc, loss = self.fwd_pass(batch_x, batch_y, train=True)

                if i % 50 == 0 and create_file:   # Runs infrequently and only if desired
                    val_acc, val_loss = self.test(test_x, test_y, size=32)
                    model_log.write(f"{self.model_name},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n")

    def fwd_pass(self, x, y, train=False):
        """Can be used for training or testing.
        *Called by train_and_test_v2
        """

        # If training, zero the gradient.
        if train:
            self.optimizer.zero_grad()

        # Run set through NN and access output (determinations)
        outputs = self.net(x)

        # Calculate accuracy rate
        matches = [torch.argmax(i) == torch.argmax(j)  for i, j in zip(outputs, y)]  # Output= [bool, bool, ...]
        accuracy = matches.count(True)/len(matches)

        # Compute the loss
        loss = self.loss_function(outputs, y)

        if train:
            loss.backward()
            self.optimizer.step()

        return accuracy, loss

    def test(self, test_x, test_y, size=32):
        """Method for testing.
        *train_and_test_v2
        """

        random_start = random.randint(0, len(test_x))-size

        # Initialize test sets
        x = test_x[random_start:random_start+size].view(-1, 1, self.spec_width, self.spec_length)
        y = test_y[random_start:random_start+size]

        # Call fwd_pass to run through model
        with torch.no_grad():
            accuracy, loss = self.fwd_pass(x.to(self.device), y.to(self.device), train=False)

        # Get accurary and loss
        return accuracy, loss

    def img_arr_to_tensor(self, arr):
        """Converts an image (mel-spectrogram) to a tensor to be given to the model.
        args: arr = image array
        returns: tensor instance of the arr"""
        data_tensor = torch.Tensor(arr)
        data_tensor = data_tensor.view(-1, self.spec_width, self.spec_length)  # Reshape tensor
        return data_tensor

    def predict_song(self, song_path):
        """Uses the model to predict the genre of a song.
        song_path: path to a song clip
        returns
        """
        pa = PrepareAudio()
        spectrogram = pa.start(song_path)
        # Check that the spectrogram transformation was successful
        if spectrogram[0] is False:
            # Return the error message from the failed spectrogram transformation
            return spectrogram[1]
        # Convert the spectrogram to a data tensor
        data_tensor = self.img_arr_to_tensor(spectrogram)
        results = self.net(data_tensor).tolist()            # Get results from the neural network
        results = results[0]                                # First result set in the results
        self.print_prediction_results(results)
        return results

    def genre_file_to_dict(self, dict_path):
        """Reads in the path to a genre dictionary and converts it into a python dictionary.
        args: dict_path = path to the .txt file containing a json object of the genre dict.
        """
        # Open json file
        file = open(dict_path)
        return json.load(file)

    def print_prediction_results(self, results):
        """Prints the prediction results using the genre dictionary."""
        for genre in self.genre_dict:
            i = self.genre_dict[genre]
            print(f'{genre}: {results[i]*100}%')
        return
