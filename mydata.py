import numpy as np
import pandas as pd
import csv
import random
from sklearn.model_selection import train_test_split

class Mydataset(object):

    def load_user_traindata(self):
        # 导入csv文件 构建训练集
        test = pd.read_table("./data/Gowalla.txt", header=None)
        users,venues = test[0],test[4]  # 根据index来取值
        a,b = list(users),list(venues)
        A = np.array(a)[:,np.newaxis]
        B = np.array(b)[:,np.newaxis]
        D = np.hstack((A,B))
        D = D[0:436991] #
        #print(D)
        header = ['userId', 'venueId']
        test = pd.DataFrame(columns=header, data=D)
        test.drop_duplicates(inplace=True)
        #ff = test.sample(frac=1)  # 随机打乱
        test.to_csv('data/gowalla/userTrain.csv', index=False)
        print("ok?")

        venueIdList = {}
        with open('data/gowalla/userTrain.csv', "r") as f:
            reader = csv.reader(f)
            for r in reader:
                if r[1] == 'venueId':
                    continue
                u, v = int(r[0]), int(r[1])
                if v not in venueIdList:
                    venueIdList[v] = 0
                venueIdList[v] = venueIdList[v] + 1
        print(len(venueIdList))

        # 删除这些场所记录
        df = pd.read_csv("data/gowalla/userTrain.csv")
        for k, v in venueIdList.items():
            if v == 1:
                df = df.drop(index=df.loc[(df['venueId'] == k)].index)  # 删除csv文件的数据
        df.to_csv('data/gowalla/userTrain.csv', index=False)
        print("删除完毕！")


        count = [0] * 1700
        with open('data/gowalla/userTrain.csv', "r") as f:
            reader = csv.reader(f)
            for r in reader:
                if r[0] == 'userId':
                    continue
                u = int(r[0])
                if u != -1:
                    count[u] = count[u] + 1
        #print(count)
        # 删除这些用户记录
        df = pd.read_csv("data/gowalla/userTrain.csv")
        for i in range(len(count)):
            if count[i] == 0:
                continue
            if count[i] < 50:
                df = df.drop(index=df.loc[(df['userId'] == i)].index)  # 删除csv文件的数据
        df.to_csv('data/gowalla/userTrain.csv', index=False)
        df = pd.read_csv("data/gowalla/userTrain.csv")
        df_new = df.sample(frac=1)  # 随机打乱
        df_new.to_csv('data/gowalla/userTrain.csv', index=False)
        print("删除用户完毕")
        #


    def transform_venueId(self):
        # 将场所ID转换成数字
        # 统计场所ID出现的次数
        print("[INFO] 开始构建字典")
        venueIdList = {}
        count = -1
        with open('data/gowalla/userTrain.csv', "r") as f:
            line = f.readline()
            line = line.rstrip()
            while line != None and line != "":
                arr = line.split(",")
                user, item = arr[0], arr[1]
                if item not in venueIdList:
                    venueIdList[item] = []
                venueIdList[item].append(count)
                line = f.readline()
                line = line.rstrip()
                count = count + 1
        print("[INFO]构建字典结束")
        print(venueIdList)
        print(len(venueIdList)) # 23332个物品,其中包含了venueId
        print("[INFO]打开CSV文件")
        # 开始替换
        df = pd.read_csv("data/gowalla/userTrain.csv")
        df['userId'] = df['userId'].astype(int)
        print("[INFO]打开文件完成")
        newName = 0
        for key, value in venueIdList.items():
            #print(f"[INFO] 修改venueId {key} 为 {newName}")
            if len(value) > 0:
                for index in value:
                    df.loc[index, 'venueId'] = newName #将对应行的场所ID改为数字
            newName = newName + 1
        print("结束")
        df['userId'] = df['userId'].fillna(-1).astype(int) #修改csv文件后保证userid依旧为整数
        df['userId'] = df['userId'].replace('-1', np.nan) #如果有空的情况就用np.nam代替，这样就可以用上面这行代码
        df['venueId'] = df['venueId'].fillna(-1).astype(int)
        df['venueId'] = df['venueId'].replace('-1', np.nan)
        df_new = df.sample(frac=1)  # 随机打乱
        df_new.to_csv('data/gowalla/userTrain.csv', index=False)


    def load_user_testdata(self):
        #从训练集里面拿出每个用户的5条记录作为测试集，如果有不够5条的怎么办，已证实每个用户有5条记录
        header = ['userId', 'venueId']
        with open('data/gowalla/userTrain.csv', "r") as f:
            reader = csv.reader(f)
            with open('data/gowalla/userTest.csv', "w", newline='') as file_obj:
                writer = csv.writer(file_obj)
                writer.writerow(header)
                # 初始化字典
                count = {}
                for i in range(1700):
                    count[i] = 0
                with open('data/gowalla/userTrainNew.csv', "w", newline='') as file_o:
                    writer2 = csv.writer(file_o)
                    writer2.writerow(header)
                    # 遍历读取文件
                    for r in reader:
                        if r[0] == 'userId':
                            continue
                        u, v = int(r[0]), int(r[1])
                        if u != -1 and v > 0:
                            if (count[u] < 5):
                                # 一行行写入新文件
                                writer.writerow(r)
                                count[u] = count[u] + 1
                            # 将剩下的写入新的训练集文件
                            else:
                                writer2.writerow(r)
        print("成功写入user测试集文件")
        # 再对训练集进行随机打乱
        df = pd.read_csv("data/gowalla/userTrainNew.csv")
        df_new = df.sample(frac=1)  # 随机打乱
        df_new.to_csv('data/gowalla/userTrainNew.csv', index=False)
        df = pd.read_csv("data/gowalla/userTest.csv")
        df_new = df.sample(frac=1)  # 随机打乱
        df_new.to_csv('data/gowalla/userTest.csv', index=False)

    def user_negative(self):
        # 统计物品的总数,生成一个包含所有物品的列表，统计每个用户交互过的物品列表
        # 这两个列表取差集，在差集里面随机生成100个物品
        venue_sum = list(range(37062))
        # venue_sum = list(range(32546))
        #print(venue_sum)
        # 用字典统计每个用户交互过的物品列表
        count = {}
        for i in range(1700):
            count[i] = []
        with open('data/gowalla/userTrain.csv', "r") as f:
            reader = csv.reader(f)
            for r in reader:
                if r[0] == 'userId':
                    continue
                u, v = int(r[0]), int(r[1])
                if u != -1 and v > 0:
                    if v not in count[u]:
                        count[u].append(v)
        #print(count)
        # 取差集,并写入负样本列表
        neg_num = {}
        for i in range(1700):
            neg_num[i] = []
        for item, value in count.items():
            neg_num[item] = list(set(venue_sum) - set(value))
            neg_num[item] = random.sample(neg_num[item], 100)
        #print(neg_num)
        print("成功统计")
        return neg_num

    def user_neg_csv(self):
        neg_num = self.user_negative()
        # 将负样本写入csv文件
        with open('data/gowalla/userTest.csv', "r") as f:
            reader = csv.reader(f)
            print("[INFO]成功读取测试集文件")
            with open('data/gowalla/userNegative.csv', "w", newline='') as file:
                wf = csv.writer(file)
                for r in reader:
                    if r[0] == 'userId':
                        continue
                    u, v = int(r[0]), int(r[1])
                    for i in neg_num[u]:
                        r.append(i)
                    wf.writerow(r)
        print("成功写入user负样本")


#############################################################################
    # group与物品的交互是其组成员与物品交互的并集
    def load_group_traindata(self,group_user):
        group_venue = []
        i = 0
        for g in group_user.values():
            if g:
                for u in g:
                    u_venue = self.find(u)
                    for v in u_venue:
                        group_venue.append([i, v])
                i = i + 1
        print(group_venue)
        # 将列表写入csv文件
        header = ['groupId', 'venueId']
        test = pd.DataFrame(columns=header, data=group_venue)
        test.drop_duplicates(inplace=True)
        ff = test.sample(frac=1)  # 随机打乱
        ff.to_csv('data/gowalla/groupTrain.csv', index=False)#这里做了修改
        print("成功写出组的训练集")

    def find(self, u):
        f = pd.read_csv("data/gowalla/userTrain.csv")
        cond = f.userId == u
        f1 = f[cond]
        list = f1.values.tolist()
        u_venue = []
        # count = 0
        for i in list:
            u_venue.append(i[1])
            # count = count + 1
        return u_venue

    # group的测试集物品应该在用户测试集中选
#########################################################################################
    def group_newtest(self,group_user):
        group_test = {}
        for i in range(621):
            group_test[i] = []
        i = 0
        for g in group_user.values():
            if g:
                for u in g:
                    ut = self.find_u(u)
                    group_test[i].extend(ut)
                group_test[i] = random.sample(group_test[i], 5)
                i = i + 1
        #print(group_test)
        return group_test

    def find_u(self, u):
        f = pd.read_csv("data/gowalla/userTest.csv")
        cond = f.userId == u
        f1 = f[cond]
        list = f1.values.tolist()
        u_venue = []
        # count = 0
        for i in list:
            u_venue.append(i[1])
            # count = count + 1
        return u_venue

    def load_group_newtest(self,group_user):
        group_test = self.group_newtest(group_user)
        # 将其写入csv文件
        with open('data/gowalla/groupTest.csv', "w", newline='') as file_obj:
            writer = csv.writer(file_obj)
            for key,value in group_test.items():
                for v in value:
                    writer.writerow([key,v])
        df = pd.read_csv("data/gowalla/groupTest.csv")
        ff = df.sample(frac=1)  # 随机打乱
        ff.to_csv('data/gowalla/groupTest.csv', index=False)
        print("成功写入新的测试集")

    def load_group_newtrain(self,group_user):
        group_test = self.group_newtest(group_user)
        with open('data/gowalla/groupTrain.csv', "r") as f:
            reader = csv.reader(f)
            with open('data/gowalla/groupTrainNew.csv', "w", newline='') as file_obj:
                writer = csv.writer(file_obj)
                for r in reader:
                    if r[0] == 'groupId':
                        continue
                    u, v = int(r[0]), int(r[1])
                    if u != -1 and v > 0:
                        if v not in group_test[u]:
                            writer.writerow(r)
        print("成功写入group新的训练集文件")

    def group_negative(self, group_user):
        # 把组成员的负样本列表合并起来，再随机选取100个作为组的负样本列表
        neg_num = self.user_negative()
        group_neg = {}
        for i in range(621):
            group_neg[i] = []
        i = 0
        for g in group_user.values():
            if g:
                for u in g:
                    if u != -1:
                        group_neg[i].extend(neg_num[u])
                group_neg[i] = random.sample(group_neg[i], 100)
                i = i + 1
        #print(group_neg)

        # 将负样本写入csv文件
        with open('data/gowalla/groupTest.csv', "r") as f:
            reader = csv.reader(f)
            print("[INFO]成功读取测试集文件")
            with open('data/gowalla/groupNegative.csv', "w", newline='') as file:
                wf = csv.writer(file)
                for r in reader:
                    if r[0] == 'groupId':
                        continue
                    u, v = int(r[0]), int(r[1])
                    for i in group_neg[u]:
                        r.append(i)
                    wf.writerow(r)
        print("成功写入group负样本")

#################################################################################


    def test(self):
        print("5555")
        venueIdList = {}
        with open('data/gowalla/userTrain.csv', "r") as f:
            reader = csv.reader(f)
            for r in reader:
                if r[1] == 'venueId':
                    continue
                u, v = int(r[0]), int(r[1])
                if v not in venueIdList:
                    venueIdList[v] = 0
                venueIdList[v] = venueIdList[v] + 1
        print(len(venueIdList))










