'''
Created on Nov 10, 2017

Deal something

@author: Lianhai Miao
'''
import torch
import numpy as np
import math
import heapq #堆
import pandas as pd
import csv


# 我懂了！这里的评估是针对测试集，然后测试集里面有一个对应的用户-物品，另外负样本里面还有这个用户的100个未交互过的物品，所以会预测101个物品的分数

class Helper(object):
    """
        utils class: it can provide any function that we need
    """
    def __init__(self):
        self.timber = True #日志？

    def gen_group_member_dict(self, path):
        """
        Load Group-Users Data
        """
        g_m_d = {}
        with open(path, 'r') as f:
            line = f.readline().strip()
            # 数据行示例：214 559,570
            while line != None and line != "":
                a = line.split(',')
                g = int(a[0])
                g_m_d[g] = []
                # for m in a[1].split(','):
                for m in a[1:]:
                    if m != '-1':
                        g_m_d[g].append(int(m)) #每个组包含的用户
                line = f.readline().strip()
        return g_m_d
    # g_m_d是一个字典，里面存了每一个群组所包含的用户，表示为群组：用户1，用户2...

    def gen_item_follow_dict(self, path):
        g_m_d = {}
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                a = line.split(',')
                g = int(a[0])
                g_m_d[g] = []
                for m in a[1:]:
                    g_m_d[g].append(int(m))
                line = f.readline().strip()
        return g_m_d

    def evaluate_model(self, model, testRatings, testNegatives, K, type_m):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.对测试集的每一个群组进行物品的预测
        """
        u_scores, hits, ndcgs = [], [], []
        compe = []
        for idx in range(len(testRatings)): #对每一个样本进行评估
            (u_score,hr,ndcg) = self.eval_one_rating(model, testRatings, testNegatives, K, type_m, idx)
            if hr == 1:
                compe.append(testRatings[1])
            u_scores.append(u_score)
            hits.append(hr)
            ndcgs.append(ndcg)
        return (u_scores,compe,hits, ndcgs)

    def eval_one_rating(self, model, testRatings, testNegatives, K, type_m, idx):
        rating = testRatings[idx] #形式为：[用户，物品]
        items = testNegatives[idx] #这里存着没有交互过的物品，是100个
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        #print(items) #证实为items列表里有101个物品

        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u) #返回一个指定形状、类型的数组。长度为len(items),数值为u
        #print(users) 是的，101个一模一样的用户数组
        users_var = torch.from_numpy(users) #该方法就是把数组转换为张量
        items_var = torch.LongTensor(items) #对应101个物品的同一个用户？
        # 开始预测
        if type_m == 'group':
            predictions = model(users_var, None, items_var)
        elif type_m == 'user':
            predictions = model(None, users_var, items_var)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.data.numpy()[i]  #每一个物品对应一个预测分数!
            #print(map_item_score[item])


        # 满意度
        u_score = predictions.data.numpy()[100]
        items.pop() #然后只剩下100个物品
        #print(items)

        # Evaluate top rank list
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem) #看这个物品在推荐列表的第几位
        return (u_score,hr, ndcg)

    def getHitRatio(self, ranklist, gtItem): #命中率？
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem): #这个是检测推荐列表的顺序的影响，越靠后NDCG越小
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0


