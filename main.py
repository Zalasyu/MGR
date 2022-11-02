import sys
import os.path
# sys.path.append('/src/data/')
from src.data.makedataset import MusicTrainingData
from src.models.train_model import Net, Model
# from makedataset import MusicTrainingData

if __name__ == "__main__":
    # Instructions: # add data_path and output_path
    
    # create spectrograms and numpy array    
    # music_training_data = MusicTrainingData()
    # data_path = 'data/raw'
    # output_path = 'data/processed'
    # music_training_data.make_training_data(data_path, output_path)

    # build model
    # net = Net()
    # print(net)
    model = Model()
    model.train_model()
