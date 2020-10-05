import tensorflow as tf

from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, history_length):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(history_length, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 32, kernel_size=5)
        # self.bn0 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 128, kernel_size=5)

        self.conv4 = nn.Conv2d(128, 512, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(512)
        # self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(2048, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = self.bn0(x)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = self.bn1(x)
        x = x.view(-1, 2048)
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)

        return x

        # TODO: Loss and optimizer
        # ...
        #
        # TODO: Start tensorflow session
        # ...

        # self.saver = tf.train.Saver()

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
