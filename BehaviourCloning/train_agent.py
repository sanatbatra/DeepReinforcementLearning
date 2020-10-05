from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import torchvision
import torch
import random
from operator import itemgetter
import torch.nn as nn

from BehaviourCloning.model import Model
from BehaviourCloning.utils import *
import math

from torch.utils.tensorboard import SummaryWriter


def read_data(datasets_dir="./data", frac = 0.1, frac_of_datapoints=1.0):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl_ekr6072.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = int(len(data["state"]) * frac_of_datapoints)
    X = X[:n_samples]
    y = y[:n_samples]
    X_train, y_train = X[:int((1 - frac) * n_samples)], y[:int((1 - frac) * n_samples)]
    X_valid, y_valid = X[int((1 - frac) * n_samples):], y[int((1 - frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)

    new_X_train = [X_train[i:i + history_length] for i in range(len(X_train) - history_length + 1)]
    new_y_train = [y_train[i + history_length - 1] for i in range(len(y_train) - history_length + 1)]

    new_X_valid = [X_valid[i:i + history_length] for i in range(len(X_valid) - history_length + 1)]
    new_y_valid = [y_valid[i + history_length - 1] for i in range(len(y_valid) - history_length + 1)]

    print('Done')
    return torch.tensor(new_X_train), torch.tensor(new_y_train), torch.tensor(new_X_valid), torch.tensor(new_y_valid)


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, history_length, model_dir="./models", tensorboard_dir="./tensorboard"):
    writer = SummaryWriter(tensorboard_dir)

    print("... train model")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    # TODO: specify your neural network in model.py
    agent = Model(history_length)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr, weight_decay=1e-5)
    loss_func = nn.MSELoss()

    # X_valid = X_valid.reshape(-1, history_length, 96, 96)
    agent.train()

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training in your web browser

    # training loop

    for i in range(n_minibatches):
        optimizer.zero_grad()
        permutation = torch.randperm(X_train.shape[0])
        indices = permutation[:batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        # batch_x = batch_x.reshape(-1, history_length, 96, 96)
        output = agent(batch_x)
        loss = loss_func(output, batch_y)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            agent.eval()
            # in case you wanted a semi-full example
            outputs = agent(X_valid)
            loss_valid = loss_func(outputs, y_valid)

            print('Train Epoch: {} \tTrain Loss: {:.6f}, Validation Loss: {:.6f}'.format(i, loss.item(),
                                                                                         loss_valid.item()))
            torch.save(agent.state_dict(), 'agent_20000.pth')
            writer.add_scalar("Training Loss", loss, i)
            writer.add_scalar("Validation Loss", loss_valid, i)
            agent.train()

    writer.flush()
    # TODO: save your agent
    # model_dir = agent.save(os.path.join(model_dir, "agent.ckpt"))

    torch.save(agent.state_dict(), os.path.join(model_dir, 'agent_20000.pth'))
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data", frac_of_datapoints=1.0)

    history_length = 5

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=history_length)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=4000, batch_size=64, lr=0.0001, history_length=history_length)
 
