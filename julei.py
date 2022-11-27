import numpy as np
import random
import pandas as pd
import csv
from copy import deepcopy
from mydata import Mydataset

class Cluster(object):
    # def __init__(self,k,p_u,s):
    #     self.k = k
    #     self.p_u = p_u
    #     self.s = s

    def group_user(self):
        group_user = self.from_group()
        header = ['user1', 'user2', 'user3', 'user4']
        with open('data/gowalla/groupMember.csv', "w", newline='') as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(header)
            for g in group_user.values():
                if g:
                    writer.writerow(g)
        # 添加组ID,从0开始
        df = pd.read_csv("data/gowalla/groupMember.csv")
        df['user1'] = df['user1'].fillna(-1).astype(int)
        df['user1'] = df['user1'].replace('-1', np.nan)
        df['user2'] = df['user2'].fillna(-1).astype(int)
        df['user2'] = df['user2'].replace('-1', np.nan)
        df['user3'] = df['user3'].fillna(-1).astype(int)
        df['user3'] = df['user3'].replace('-1', np.nan)
        df['user4'] = df['user4'].fillna(-1).astype(int)
        df['user4'] = df['user4'].replace('-1', np.nan)
        # df_new = df.sample(frac=1)  # 随机打乱
        df.to_csv('data/gowalla/groupMember.csv')
        print("成功添加组id,并写入文件")
        return group_user

    def sensor_random(self):
        # 初始化每个用户所拥有的传感器为空列表
        user_sensor = {}
        r = []
        with open('data/gowalla/userTrain.csv', "r") as f:
            reader = csv.reader(f)
            for a in reader:
                if a[1] == 'venueId':
                    continue
                u, v = int(a[0]), int(a[1])
                if u not in r:
                    r.append(u)
        print(len(r))  # 576
        print(max(r))

        for i in r:
            user_sensor[i] = []  #用户ID从1开始
        p_u = 0.45 #用户的传感器接受概率
        s = 7 #传感器总数
        for i in r:
            for j in range(s):
                x = np.random.rand() #产生一个0-1之间的数，服从0-1之间的均匀分布
                if x < p_u:
                    user_sensor[i].append(j)
            if user_sensor[i] == []:   #可能会存在空列表,必须保证每个用户至少拥有一个传感器
                x = random.randint(0, 6)  # 返回这之间的随机数
                user_sensor[i].append(x)
            if len(user_sensor[i]) == 7:
                y = random.randint(0, 6)  # 返回这之间的随机数
                user_sensor[i].remove(y)
        print(user_sensor)

        dict = deepcopy(user_sensor)
        with open('data/gowalla/userSensor.csv', "w", newline='') as file_obj:
            writer = csv.writer(file_obj)
            for key, value in dict.items():
                if value != []:
                    value.insert(0, key)
                    writer.writerow(value)
        print("写入用户传感器")
        return user_sensor

    def from_group(self):
        user_sensor = self.sensor_random()
        aa = user_sensor.keys()
        if -1 in aa:
            del user_sensor[-1]

        # 开始聚类
        group_k = {}
        for i in range(1, 500):
            group_k[i] = []
        os = []
        k = 1
        while (len(user_sensor.items()) > 1):
            i = random.sample(user_sensor.keys(), 1)[0]
            v = user_sensor[i]

            group_k[k].append(i)
            while len(group_k[k]) < 4 and v != [0, 1, 2, 3, 4, 5, 6]:
                sim_max = 1.1
                op = 1700
                for j, u in user_sensor.items():
                    if j in group_k[k] or j == -1:
                        continue
                    else:
                        inter = list(set(v) & set(u))  # 求交集
                        union = list(set(v) | set(u))  # 求并集
                        if len(union) > 0:
                            si = len(inter) / len(union)
                            if si < sim_max:
                                sim_max = si
                                op = j
                                os = u.copy()
                if op != 1700:
                    group_k[k].append(op)
                    v.extend(os)
                    v = list(set(v))  # 去重

            for a in group_k[k]:
                del user_sensor[a]

            print(f"k is {k}, and group {group_k[k]}, v is {v}")
            k = k + 1

            if len(user_sensor) == 1:
                group_k[k-1].append(user_sensor.keys())
                break
        return group_k

    def itemSensor(self):
        user_sensor = self.sensor_random()
        venueIdList = {}
        with open('data/gowalla/userTrain.csv', "r") as f:
            reader = csv.reader(f)
            for r in reader:
                if r[1] == 'venueId':
                    continue
                u, v = int(r[0]), int(r[1])
                if v not in venueIdList:
                    venueIdList[v] = []
                venueIdList[v].append(u)
        print(len(venueIdList))

        sensor_venue = {}
        for k in venueIdList.keys():
            sensor_venue[k] = []
        p_t = 0.6
        # 生成任务的传感器
        for k, v in venueIdList.items():
            if len(v) == 1:
                # for j in range(7):
                    # x = np.random.rand()  # 产生一个0-1之间的数，服从0-1之间的均匀分布
                    # if x < p_t:
                    #     sensor_venue[k].append(j)
                sensor_venue[k].extend(user_sensor[v[0]])
            elif len(v) == 2:
                for u in v:
                    sensor_venue[k].extend(user_sensor[u])
            else:
                v = random.sample(v, 2)
                for u in v:
                    sensor_venue[k].extend(user_sensor[u])
            sensor_venue[k] = list(set(sensor_venue[k]))

        for k,v in sensor_venue.items():
            if len(v) == 1:
                s = [0,1,2,3,4,5,6]
                w = list(set(s) - set(v))
                x = random.sample(w, 1)[0]
                v.append(x)
            if len(v) == 7:
                y = random.randint(0, 6)  # 返回这之间的随机数
                v.remove(y)
        # 写入csv文件
        with open('data/gowalla/sensor.csv', "w", newline='') as file_obj:
            writer = csv.writer(file_obj)
            for k, v in sensor_venue.items():
                v.insert(0, k)
                writer.writerow(v)
        df = pd.read_csv("data/gowalla/groupTest.csv")
        ff = df.sample(frac=1)  # 随机打乱
        ff.to_csv('data/gowalla/groupTest.csv', index=False)
        print("成功写入任务传感器")

