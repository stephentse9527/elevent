'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)

Modified  on Nov 10, 2017, by Lianhai Miao
'''

#这个dataset里面的训练矩阵拿来作为训练集参加训练，而测试列表和负样本列表都是用在评估里面

import scipy.sparse as sp #稀疏矩阵，scipy包的sparse模块实现数据存储
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import csv

class GDataset(object):

    def __init__(self, user_path, group_path, num_negatives):
        '''
        Constructor
        '''
        self.num_negatives = num_negatives #6个负样本
        # user data
        self.user_trainMatrix = self.load_rating_file_as_matrix(user_path + "TrainNew.csv")
        self.user_testRatings = self.load_rating_file_as_list(user_path + "Test.csv")
        self.user_testNegatives = self.load_negative_file(user_path + "Negative.csv")
        self.num_users, self.num_items = self.user_trainMatrix.shape #矩阵行表示用户数，列表示物品数
        # group data
        self.group_trainMatrix = self.load_rating_file_as_matrix(group_path + "TrainNew.csv")
        self.group_testRatings = self.load_rating_file_as_list(group_path + "Test.csv")
        self.group_testNegatives = self.load_negative_file(group_path + "Negative.csv")

    # 矩阵，行表示用户，列表示物品，有用户-物品交互的话，对应元素为1
    def load_rating_file_as_matrix(self, filename):
        # 获取用户和物品的数目
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(",")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i) #取最大值
                line = f.readline()

        # Construct matrix
        # num_users, num_items 为用户和物品编号的最大值，这里 +1 是因为矩阵的索引从[0,0]开始，有行为统计则rate为1
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32) #dok_matrix适合逐渐添加元素:将非零值保存在字典，非零值的坐标元组作为字典的键
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(",")
                # if len(arr) > 2:
                #     user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                #     if (rating > 0): #因为有评分为0的情况
                #         mat[user, item] = 1.0
                # else:
                #     user, item = int(arr[0]), int(arr[1])
                #     mat[user, item] = 1.0 #为什么还是1？ ?????
                user, item = int(arr[0]), int(arr[1])
                mat[user, item] = 1.0
                line = f.readline()
        return mat

    # <用户,物品>对
    def load_rating_file_as_list(self, filename): #因为测试集没有评分，所以列表足够？
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(",")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item]) #列表里面放列表
                line = f.readline()
        return ratingList

    # 该列表存储所有的负样本
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(",")
                negatives = []
                # for x in arr[1:]:
                for x in arr[2:]:
                    negatives.append(int(x))
                negativeList.append(negatives) #格式为：[[100个],[100个],...]
                line = f.readline()
        return negativeList

    #得到训练样本  这里的负样本是从矩阵中不为1的值所对应的列（物品）得来的
    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input = [], [], []
        (num_users, num_items) = train.shape  # train是train_matrix
        # 是所有值为1对应的物品 交互过的物品-6个没交互过的物品 可是没有交互过的物品为什么做训练集呢？他为什么会有标签呢？
        for (u, i) in train.keys():  # 矩阵中所有值为1所对应的用户-物品ID
            # print(train[u,i])
            # 正例 之所以采样 num_negatives 次，是因为后边zip时，组成num_negatives个 正负采样 组合
            for _ in range(self.num_negatives):
                pos_item_input.append(i)  # 6个一模一样的正例item
            # 负例
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)  # 为了对应：6个用户，6个正负例
                neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]  # 1个正例1个负例
        return user_input, pi_ni

    #得到用户的数据集，小批量
    def get_user_dataloader(self, batch_size):
        user, positem_negitem_at_u = self.get_train_instances(self.user_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(positem_negitem_at_u)) #变成张量 变成一个用户对应一个正负例
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) #变成特征和带标签的数据集,标签是正负例
        return user_train_loader

    # 就是用TensorDataset把用户和正负例合起来变成一个样例，格式为（用户，[正例，负例]）
    # 然后用dataLoader加载数据集，里面有数据和标签，相当于数据是用户，标签是正负例？？？！！！
    def get_group_dataloader(self, batch_size):
        group, positem_negitem_at_g = self.get_train_instances(self.group_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_negitem_at_g))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        #for i, data in enumerate(group_train_loader, 1):  # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
        #    x_data, label = data
        #    print(' batch:{0} x_data:{1}  label: {2}'.format(i, x_data, label))
        # 为什么batch会上万呢，可能是因为在构建正负例的时候，一个用户ID会复制6份
        return group_train_loader





