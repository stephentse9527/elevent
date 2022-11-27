'''
Created on Nov 10, 2017
Create Model

@author: Lianhai Miao
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AGREE(nn.Module):
    def __init__(self, num_users, num_items, num_groups, num_follow, embedding_dim, group_member_dict, item_follow_dict,
                 drop_ratio):
        super(AGREE, self).__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        self.followembeds = FollowEmebddingLayer(num_follow, embedding_dim)

        self.followAttention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        self.group_member_dict = group_member_dict
        # user,fans dict
        self.item_follow_dict = item_follow_dict
        # 　
        self.num_follow = num_follow
        self.num_users = num_users
        self.num_groups = num_groups

        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)


    def forward(self, group_inputs, user_inputs, item_inputs):
        # train group
        if (group_inputs is not None) and (user_inputs is None):
            out = self.grp_forward(group_inputs, item_inputs)
        # train user
        else:
            out = self.usr_forward(user_inputs, item_inputs)
        return out

    # group forward
    def grp_forward(self, group_inputss, item_inputss):
        group_embeds = Variable(torch.Tensor())
        item_embeds_full = self.item_aggregate(item_inputss.numpy()) #这里需要改！
        group_inputs, item_inputs = self.tensor2np(group_inputss), self.tensor2np(item_inputss)
        for i, j in zip(group_inputs, item_inputs):
            members = self.group_member_dict[i]
            members_embeds = self.userembeds(Variable(torch.LongTensor(members)))  # 这里不一样
            items_numb = []
            for _ in members:
                items_numb.append(j)
            item_embeds = self.item_aggregate(items_numb)
            group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)  # 要与item连接到一起
            at_wt = self.attention(group_item_embeds)
            g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
            group_embeds_pure = self.groupembeds(Variable(torch.LongTensor([i])))
            g_embeds = g_embeds_with_attention + group_embeds_pure
            if group_embeds.dim() == 0:
                group_embeds = g_embeds
            else:
                group_embeds = torch.cat((group_embeds, g_embeds))

        element_embeds = torch.mul(group_embeds, item_embeds_full)  # Element-wise product
        new_embeds = torch.cat((element_embeds, group_embeds, item_embeds_full), dim=1)
        y = F.sigmoid(self.predictlayer(new_embeds))
        return y

    def item_aggregate(self, item_inputs):
        item_finnal_list = []
        for i in item_inputs:
            follows = self.item_follow_dict[i]
            follow_embeds = self.followembeds(Variable(torch.LongTensor(follows)))
            items_numb = len(follows)
            # user embedding
            item_embeds = self.itemembeds(Variable(torch.LongTensor([i] * items_numb)))
            item_follow_embeds = torch.cat((follow_embeds, item_embeds), dim=1)
            at_wt = self.followAttention(item_follow_embeds)
            i_embeds_with_attention = torch.matmul(at_wt, follow_embeds)
            item_embeds_pure = self.itemembeds(Variable(torch.LongTensor([i])))
            u_embeds = i_embeds_with_attention + item_embeds_pure
            item_finnal_list.append(u_embeds.view(-1))
        user_finnal_vec = torch.stack(item_finnal_list, dim=0)  # 返回一个成员的表征？维度还是32维
        return user_finnal_vec

    # user forward
    def usr_forward(self, user_inputs, item_inputs):
        user_inputs_var = Variable(user_inputs)
        user_embeds = self.userembeds(user_inputs_var)
        item_embeds = self.item_aggregate(item_inputs.numpy())
        element_embeds = torch.mul(user_embeds, item_embeds)  # Element-wise product
        new_embeds = torch.cat((element_embeds, user_embeds, item_embeds), dim=1)
        y = F.sigmoid(self.predictlayer(new_embeds))
        return y

    def tensor2np(self, tens):
        return tens.numpy()

# embedding_dim 就是嵌入向量的维度，即用embedding_dim值的维数来表示一个基本单位。
# num_embeddings就是生成num_embeddings个嵌入向量。

class UserEmbeddingLayer(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim) #具有一个权重，形状是num_user个，每个embedding_dim维

    def forward(self, user_inputs):
        user_embeds = self.userEmbedding(user_inputs) #user_inputs已经是张量形式，最后形状变为输入形状×embedding_dim
        return user_embeds # user_inputs×1×embedding_dim？


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_inputs):
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds

class FollowEmebddingLayer(nn.Module):
    def __init__(self, num_follow, embedding_dim):
        super(FollowEmebddingLayer, self).__init__()
        self.followEmbedding = nn.Embedding(num_follow, embedding_dim)

    def forward(self, follow_inputs):
        follow_embeds = self.followEmbedding(follow_inputs)
        return follow_embeds

class GroupEmbeddingLayer(nn.Module):
    def __init__(self, number_group, embedding_dim):
        super(GroupEmbeddingLayer, self).__init__()
        self.groupEmbedding = nn.Embedding(number_group, embedding_dim)

    def forward(self, num_group):
        group_embeds = self.groupEmbedding(num_group)
        return group_embeds


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16), #输入2*32，隐藏层16，输出1
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        weight = F.softmax(out.view(1, -1), dim=1)
        return weight


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8), #输入3*32维，隐藏层8，输出1
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

