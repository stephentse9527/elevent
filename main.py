"""
Created on Nov 10, 2017
Main function

@author: Lianhai Miao
"""

from model.agree import AGREE
import torch
import torch.optim as optim
import numpy as np
from time import time
from config import Config
from utils.util import Helper
from dataset import GDataset
#import matplotlib.pyplot as plt
from julei import Cluster
from mydata import Mydataset

# 训练模型
def training(model, train_loader, epoch_id, config, type_m):
    # user trainning
    global pos_prediction, neg_prediction
    learning_rates = config.lr
    #根据迭代的次数进行学习率的衰减
    lr = learning_rates[0]
    if epoch_id >= 15 and epoch_id < 25:
        lr = learning_rates[1]
    elif epoch_id >=20:
        lr = learning_rates[2]
    #lr decay
    if epoch_id % 5 == 0:
        lr /= 2
    # 定义优化器      应该是梯度下降，计算梯度然后会更新参数
    optimizer = optim.RMSprop(model.parameters(), lr)
    # 数据集的形状是：用户，正负例。用户是特征，正负例是标签
    losses = []
    for batch_id, (u, pi_ni) in enumerate(train_loader):
        # Data Load
        user_input = u
        pos_item_input = pi_ni[:, 0] #第一列是正样本，第二列是负样本
        neg_item_input = pi_ni[:, 1]
        # Forward
        if type_m == 'user': #某个用户的100个正例的预测值和100个负例的预测值
            pos_prediction = model(None, user_input, pos_item_input) #调用agree里面的forward函数，可能根据参数自动匹配对应函数?
            neg_prediction = model(None, user_input, neg_item_input)
        elif type_m == 'group':
            pos_prediction = model(user_input, None, pos_item_input)
            neg_prediction = model(user_input, None, neg_item_input)
        # Zero_grad
        model.zero_grad()
        # Loss
        # 正样本的预测分数尽可能接近1，负样本的预测分数尽可能接近0，再-1，所以得到的损失就会接近于0
        loss = torch.mean((pos_prediction - neg_prediction -1) **2)
        # record loss history
        losses.append(loss)  
        # Backward
        loss.backward()
        optimizer.step()
    print('Iteration %d, loss is [%.4f ]' % (epoch_id, torch.mean(torch.stack(losses), 0)))

def evaluation(model, helper, testRatings, testNegatives, K, type_m):
    model.eval()
    (u_scores,compe, hits, ndcgs) = helper.evaluate_model(model, testRatings, testNegatives, K, type_m)
    a = []
    for i in testRatings:
        if i[1] not in a:
            a.append(i[1])
    c = 0.0
    c = len(compe) / len(a)
    score, hr, ndcg = np.array(u_scores).mean(),np.array(hits).mean(), np.array(ndcgs).mean()
    return score, c, hr, ndcg


if __name__ == '__main__':
    # 初始化参数
    config = Config()

    # #聚类,创建群组
    #p = Cluster() #需要实例化类，创建一个对象p
    # group_user = p.group_user()
    #p.from_group()
    # p.itemSensor()

    # D = Mydataset() #创建类的实例
    # D.load_user_traindata()
    # D.transform_venueId()
    # D.load_user_testdata()
    # D.user_neg_csv()
    # D.load_group_traindata(group_user)
    # D.load_group_newtest(group_user)
    # D.load_group_newtrain(group_user)
    # D.group_negative(group_user)
    # D.test()

    #初始化工具函数
    helper = Helper()

    # 加载群组内的用户，构成dict
    # {groupid: [uid1, uid2, ..], ...}
    g_m_d = helper.gen_group_member_dict(config.user_in_group_path)

    # get the dict of follow in user
    i_f_d = helper.gen_item_follow_dict(config.follow_in_user_path)

    # 初始化数据类
    dataset = GDataset(config.user_dataset, config.group_dataset, config.num_negatives)

    #X = dataset.get_group_dataloader(7)
    #for i_batch, batch_data in enumerate(X):
     #   print(i_batch)  # 打印batch编号
     #   print(batch_data[0].size())  # 打印该batch里面src
     #   print(batch_data[1].size())  # 打印该batch里面trg

    # 获取群组的数目、训练集中用户的数目、训练集中物品的数目
    num_group = len(g_m_d)
    num_users, num_items = dataset.num_users, dataset.num_items
    print("num_group is: " + str(num_group))
    print("num_users is: " + str(num_users))
    print("num_items is: " + str(num_items))

    # 训练 AGREE 模型
    # 这里只需要把用户个数物品个数和组个数传进去就可以得到每个的向量表示，因为用户物品组的ID都是连续的？从0开始
    agree = AGREE(num_users, num_items, num_group, config.num_follow, config.embedding_size, g_m_d, i_f_d, config.drop_ratio)
    #helper.eval_one_rating(agree, dataset.user_testRatings, dataset.user_testNegatives, config.topK, 'user', 1)
    # 打印配置信息
    print("AGREE 的Embedding 维度为: %d, 迭代次数为: %d, NDCG、HR评估选择topK: %d" %(config.embedding_size, config.epoch, config.topK))

    # 训练模型
    for epoch in range(config.epoch):
        agree.train() #？
        # 开始训练时间
        t1 = time()

        training(agree, dataset.get_user_dataloader(config.batch_size), epoch, config, 'user')

        training(agree, dataset.get_group_dataloader(config.batch_size), epoch, config, 'group')
        print("user and group training time is: [%.1f s]" % (time()-t1))
        # 评估模型
        t2 = time()
        score,c, u_hr, u_ndcg = evaluation(agree, helper, dataset.user_testRatings, dataset.user_testNegatives, config.topK, 'user')
        print('User Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]' % (
            epoch, time() - t1, u_hr, u_ndcg, time() - t2))
        #print('个人满意度为：%.4f' % (score))
        print('任务完成率为：%.4f' % (c))
        score,c, hr, ndcg = evaluation(agree, helper, dataset.group_testRatings, dataset.group_testNegatives, config.topK, 'group')
        print(
            'Group Iteration %d [%.1f s]: HR = %.4f, '
            'NDCG = %.4f, [%.1f s]' % (epoch, time() - t1, hr, ndcg, time() - t2))
        #print('群组满意度为：%.4f' % (score))
        print('任务完成率为：%.4f' % (c))











