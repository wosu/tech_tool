'''
PLE实现思考：
1.输入特征，连续特征和离散特征；对连续特征进行embedding
2.每个专家的DNN结构可以不同，但是本代码只是为了简单实现。故统一保持结构一样。同时
每各CGC层中，每个专家的输出维度大小一样
3.任务数量通过参数传递，共享网络专家数量和任务专家网络通过参数传递
4.堆叠基层CGC结构，通过参数设置；
5.每个专家输出维度相同,每个专家的输入一样
6.每个任务专家网络门控网络输出数量=共享专家数量+每个任务专家数量
    任务专家门控针对共享专家和自己的任务专家进行加权进行专家选择
7.共享层专家网络门控网络输出数量=共享专家数量+所有任务专家数量
    共享专家门控针对所有专家的信息进行加权进行选择
8.下一层CGC结构输入=上一层输出：共享专家数量*输出维度+所有任务专家数量*输出维度
9.最后一层CGC中没有共享网络专家的门控
10.PLE中CGC的每层输出是将所有任务网络和共享网络输出相加，还是concate
'''
import math

import torch
from torch import nn

from python.pytorch.layers.mlp import MLP


class PLE(nn.Module):
    '''
    这里为了方便，每层的ple网络中每个task的专家数量一样
    '''
    def __init__(self,dis_feas_size:list,dis_fea_num:int,fea_num,
                 num_tasks,task_expert_num_list,tasks_loss_weight,tasks_gamma,
                 share_net_expert_nums,out_dims,
                 ple_num,dnn_hidden_unit=[64,64],embedding_size=32):
        self.num_tasks = num_tasks
        self.task_expert_num_list = task_expert_num_list
        self.dnn_hidden_unit = dnn_hidden_unit
        self.ple_num = ple_num
        self.dis_feas_size = dis_feas_size
        self.dis_fea_num = dis_fea_num
        self.fea_num = fea_num

        self.dis_feas_embeds_table = nn.ModuleList()
        # 第一层中，每个task的门控是一个线性转换
        for fea_size in dis_feas_size:
            # 这里门控网络的输入，其实可以共享离散特征的embedding
            self.dis_feas_embeds_table.append(nn.Embedding(fea_size,embedding_size))
        # 计算每一层ple的输入和输出
        self.ple = nn.ModuleList()
        first_input_dims = self.dis_fea_num * embedding_size + self.fea_num-self.dis_fea_num

        # num_tasks, task_expert_num_list,
        # share_net_expert_nums, input_dims, out_dims, dnn_hidden_unit
        self.ple.append(ExtractNet(num_tasks=num_tasks,task_expert_num_list=task_expert_num_list,
                                   share_net_expert_nums=share_net_expert_nums,input_dims=first_input_dims,out_dims=out_dims,dnn_hidden_unit=dnn_hidden_unit))
        extract_net_out_dims = out_dims*(num_tasks + 1)
        for i in range(1,ple_num):
            self.ple.append(ExtractNet(num_tasks,task_expert_num_list,share_net_expert_nums,extract_net_out_dims,dnn_hidden_unit))

        self.out_cgc_net = ExtractNet(num_tasks,task_expert_num_list,share_net_expert_nums,extract_net_out_dims,dnn_hidden_unit)

        # 每个task最后输出定义
        self.tasks_layer = nn.ModuleList()
        for i in range(num_tasks):
            self.tasks_layer(MLP())

        self.task_loss_func = nn.BCELoss()
        # 每个任务的初始权重,以及初始的gamma
        self.tasks_loss_weight = tasks_loss_weight
        self.tasks_gamma = tasks_gamma

    def forward(self,X):
        '''
        :param X: 样本特征的输入，开始为离散特征，后面为连续特征
        :return:
        '''
        dis_feas_embed_weight = []
        #查找离散特征embedding表，拼接离散特征
        for i in range(self.dis_fea_num):
            # 离散特征index shape:(1,batch_size)
            dis_feas_embed_weight.append(self.dis_feas_embeds_table[i](X[:,i].long())) # shape:(batch_size,embed_size)
        # input_feas shape:(batch_size,dis_fea_size*embed_size+conti_fea_num)
        input_feas = torch.cat([torch.cat(dis_feas_embed_weight,dim=1),X[:,self.dis_fea_num:]],dim=1)
        extract_net_outs = []
        extract_net_outs.append(self.ple[0](input_feas))
        for i in range(1,self.ple_num):
            extract_net_outs.append(
                self.ple[i](extract_net_outs[i-1]))


        final_cgc_out = self.out_cgc_net(extract_net_outs[-1]) #(bs,num_tasks,out_dim)
        outs = []
        for i,task_net in enumerate(self.tasks_layer):
            # task_net(final_cgc_out[:,i,:].squeeze() shape:(bs,out_dim)
            # out shape:(batch_size,1)
            outs.append(task_net(final_cgc_out[:,i,:].squeeze()))
        return torch.stack(outs,dim=1) # (bs,num_task.1)

    def loss(self,Y,Y_pred,Y_task_label,epoch):
        '''

        :param Y: (batch_size,num_task)
        :param Y_pred: (batch_size,num_task)
        :param Y_task_label: (batch_size,num_task),标注样本是否属于某个task的样本空间
        :return:
        '''
        task_losses = 0
        for i in range(self.num_tasks):
            # 获取第i各任务的预测值和真实值,以及样本的task label
            Y_task_i = Y[:,i]
            Y_task_i_pred = Y_pred[:,i]
            Y_task_i_label = Y_task_label[:,i]
            # 计算损失，先使用样本的任务label进行0/1标识
            single_task_loss = self.task_loss_func.forward(Y_task_i_pred*Y_task_i_label,Y_task_i*Y_task_i_label)
            single_task_loss = single_task_loss.item() * (1/torch.sum(Y_task_i_label).item())

            task_batch_sample_rate = torch.sum(Y_task_i_label).item()/Y.shape[0]
            self.tasks_gamma[i] = self.tasks_gamma[i]*(math.pow(task_batch_sample_rate,epoch))
            self.tasks_loss_weight[i] = self.tasks_loss_weight[i] * self.tasks_gamma[i]
            task_losses +=single_task_loss*self.tasks_loss_weight[i]
        return task_losses

class ExtractNet(nn.Module):
    '''
    有多少个task，每个task中有多少个自己的专家
    共享网络中的专家有多少个。
    num_tasks：多少个目标任务
    task_expert_num_list：每个目标任务中，对于自己的task专家网路中有多少个专家
    input_dims：CGC网络的输入维度大小
    out_dims:每个专家输出维度大小
    dnn_hidden_unit：这里为了实现方便，让每个专家网络的隐层结构一样
    '''
    def __int__(self,num_tasks,task_expert_num_list,
                share_net_expert_nums,input_dims,out_dims,dnn_hidden_unit=[64,64],has_share_net_gate=True):
        super(ExtractNet,self).__init__()
        self.tasks_net_list = nn.ModuleList()
        self.task_net_gate_list = nn.ModuleList()
        self.share_net = ExpertNet(share_net_expert_nums,input_dims,out_dims,dnn_hidden_unit)
        self.num_tasks = num_tasks
        self.has_share_net_gate = has_share_net_gate
        for i in range(num_tasks):
            self.tasks_net_list.append(ExpertNet(task_expert_num_list[i],input_dims,out_dims,dnn_hidden_unit))
            task_i_gate_out_dims = task_expert_num_list[i]+share_net_expert_nums
            self.task_net_gate_list.append(GateNet(input_dims,task_i_gate_out_dims))

        share_gate_out_dims = sum(task_expert_num_list)+share_net_expert_nums
        self.share_net_gate = GateNet(input_dims,share_gate_out_dims)

    def forward(self,X):
        # 共享专家网络计算,share_net_out为数组
        share_net_out = self.share_net(X)
        # 各任务网络计算
        all_task_weight_out = []
        all_task_out = []
        for i in range(self.num_tasks):
            # 每个task计算结束后，使用task门控对自己的task网络和共享网络进行加权
            # task_i_out:(batch_size,任务专家量+共享专家量,output_dim)
            # task_i_gate_out:(bs,1,任务专家量+共享专家量)
            task_i_out = torch.stack(self.tasks_net_list[i](X)+share_net_out,dim=1)
            task_i_gate_out = self.task_net_gate_list[i](X)
            all_task_out.append(task_i_gate_out)

            # 使用门控对专家进行筛选，即加权求和 task_i_weight_out:(bs,output_dim)
            task_i_weight_out = torch.bmm(task_i_gate_out,task_i_out).squeeze()
            all_task_weight_out.append(task_i_weight_out)

        if self.has_share_net_gate:
            # share gate对所有专家加权 sum(task_expert_num_list)+share_net_expert_nums
            # all_experts:(bs,all_expert_num,output_dim)  (batch_size,1,expert_num)
            share_gate_out = self.share_net_gate(X)
            all_experts = torch.stack(all_task_out+share_net_out,dim=1)
            # share_net_weight_out:(bs,output_dim)
            share_net_weight_out = torch.bmm(share_gate_out,all_experts).squeeze()
            # (bs,all_experts_num*out_dims)
            return torch.cat(all_task_weight_out+share_net_weight_out,dim=1)
        else:
            #(bs,num_tasks,out_dim)
            return torch.stack(all_task_weight_out,dim=1)


class ExpertNet(nn.Module):
    '''
    专家网络
    num_experts：专家数量
    input_dim：专家层接收的输入大小，一般为特征数量或者上一层所有的专家数量*专家输出维度
    expert_hidden_units:DNN隐藏层结构
    '''
    def __int__(self,num_experts,input_dim,output_dim,expert_hidden_units=[64,64]):
        self.expert_nets = nn.ModuleList()
        for i in num_experts:
            self.expert_nets.append(MLP(input_dim,output_dim,output_activation=None))

    def forward(self,X):
        # X:(batch_size,input_dim) out:(batch_size,num_experts,output_dim)
        # return torch.stack([expert(X) for expert in self.expert_nets],dim=1)
        return [expert(X) for expert in self.expert_nets]



class GateNet(nn.Module):
    '''
    门控网络接收专家信息输入
    '''
    def __int__(self,input_dim,output_dim):
        self.gate = nn.Linear(input_dim,output_dim)

    def forward(self,X):
        '''

        :param X: nn.Linear(X,out_dim).softmax(dim=1)
        :return:
        '''
        # X shape:(batch_size,1,expert_num)
        return self.gate(X).softmax(dim=1).unsqueeze(dim=1)
