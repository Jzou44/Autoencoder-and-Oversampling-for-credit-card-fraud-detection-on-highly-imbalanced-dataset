import numpy as np
import pandas as pd
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE


class DataManager:
    def __init__(self):
        self.init_dataset()

    def init_dataset(self):
        file_path = "creditcard.csv"
        dataset = pd.read_csv(file_path)
        dataset = dataset.dropna()
        self.feature = dataset.drop(["Time", "Class"], axis=1).values
        self.target = dataset['Class'].values
        self.target = np.expand_dims(self.target, axis=1)
        # normalize dataset
        self.feature = preprocessing.scale(self.feature)
        # train:test = 4:1
        train_dataset_size = int(4 / 5 * np.shape(self.target)[0])
        self.train_feature = self.feature[:train_dataset_size, :]
        self.train_target = self.target[:train_dataset_size, :]
        self.test_feature = self.feature[train_dataset_size:, :]
        self.test_target = self.target[train_dataset_size:, :]
        # get balanced train dataset by SMOTE
        self.get_balanced_train_dataset_by_SMOTE()
        # id array for ramdom select
        self.train_dataset_index = np.arange(0, np.shape(self.train_target)[0], 1)

    def get_balanced_train_dataset_by_SMOTE(self):
        # get balanced train dataset by SMOTE
        sm = SMOTE()
        self.train_feature, self.train_target = sm.fit_sample(self.train_feature, self.train_target[:, 0])
        self.train_target = np.expand_dims(self.train_target, axis=1)

    def next_train_batch_random_select(self, batch_size=200):
        select_ids = np.random.choice(self.train_dataset_index, batch_size, replace=False)
        feature_batch = self.train_feature[select_ids, :]
        feature_batch = self.add_gaussian_noise(feature_batch)
        target_batch = self.train_target[select_ids, :]
        target_batch = self.change_batch_y(target_batch)
        return feature_batch, target_batch

    def get_all_test_dataset(self):
        feature_batch = self.test_feature
        target_batch = self.test_target
        target_batch = self.change_batch_y(target_batch)
        return feature_batch, target_batch

    def change_batch_y(self, batch_y):
        batch_y = batch_y[:, 0]
        batch_y = np.array([batch_y == 0, batch_y == 1], dtype=np.float32)
        batch_y = np.transpose(batch_y)
        return batch_y

    def add_gaussian_noise(self, nparray):
        noise = np.random.normal(loc=0.0, scale=0.02, size=np.shape(nparray))
        nparray += noise
        return nparray
