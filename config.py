'''
Created on Nov 10, 2017
Store parameters

@author: Lianhai Miao
'''

class Config(object):
    def __init__(self):
        self.path = './data/gowalla/'
        self.user_dataset = self.path + 'user'
        self.group_dataset = self.path + 'group'
        self.user_in_group_path = "./data/gowalla/groupMember.csv"
        self.follow_in_user_path = "./data/gowalla/sensor.csv"
        self.embedding_size = 32 #嵌入层的维度设置为32维
        self.epoch = 25#30
        self.num_negatives = 6
        self.batch_size = 128 #貌似用128效果更好
        # self.lr = [0.000005, 0.000001, 0.0000005]
        self.lr = [0.01, 0.01, 0.001] #效果还不错
        self.drop_ratio = 0.1
        self.topK = 10
        self.num_follow = 7
