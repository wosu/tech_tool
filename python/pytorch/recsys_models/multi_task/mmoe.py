import torch
from torch import nn

from python.pytorch.layers.mlp import MLP


class MMoE(nn.Module):
    def __init__(self,dis_feas_size:list,dis_fea_num:int,fea_num,expert_num,
                 task_num,expert_out_dim,experts_hidden_units,tasks_hidden_units,
                 task_loss_weights,embedding_size=32):
        self.dis_fea_num = dis_fea_num
        self.task_loss_weights = task_loss_weights

        self.gate_nets = []
        self.expert_nets = []
        self.task_nets = []
        self.conti_fea_num = fea_num - dis_fea_num
        expert_input_dims = dis_fea_num*embedding_size + self.conti_fea_num
        gate_input_dims = fea_num

        for i in range(expert_num):
            self.expert_nets.append(MLP(expert_input_dims,expert_out_dim,hidden_units=experts_hidden_units))
        for i in range(task_num):
            self.gate_nets.append(nn.Linear(gate_input_dims,expert_num))
            self.task_nets.append(MLP(expert_out_dim,1,hidden_units=tasks_hidden_units))

        self.linear_dis_fea_embeds_table = nn.ModuleList()
        self.deep_dis_fea_embeds_table = nn.ModuleList()
        for fea_size in dis_feas_size:
            # 这里门控网络的输入，其实可以共享离散特征的embedding
            self.linear_dis_fea_embeds_table.append(nn.Embedding(fea_size,1))
            self.deep_dis_fea_embeds_table.append(nn.Embedding(fea_size,embedding_size))

    def forward(self,X):
        gate_tmp_inputs = []
        experts_tmp_inputs = []
        for i in range(self.dis_fea_num):
            fea_idx = X[:,i].long()
            # (batch_size,1)  (batch_size,embdding_size)
            gate_tmp_inputs.append(self.linear_dis_fea_embeds_table[i](fea_idx))
            experts_tmp_inputs.append(self.deep_dis_fea_embeds_table[i](fea_idx))

        experts_inputs = torch.cat([torch.cat(experts_tmp_inputs,dim=1),X[:,self.dis_fea_num:]],dim=1)
        gate_inputs = torch.cat([torch.cat(gate_tmp_inputs,dim=1),X[:,self.dis_fea_num:]],dim=1)

        expert_nets_out = []
        for expert_net in self.expert_nets:
            #expert_net out:(batch_size,expert_out_dim)
            expert_nets_out.append(expert_net(experts_inputs))
        expert_nets_out = torch.stack(expert_nets_out,dim=1) # (batch_size,num_experts,dim)

        task_outs = []
        for i,task_net in enumerate(self.task_nets):
            # gate_i:(batch_size,1,num_experts)  expert_nets_out:(batch_size,num_experts,dim)
            gate_i = self.gate_nets[i](gate_inputs)[:,i].softmax(dim=1).unsqueeze(dim=1)
            # task_input:(batch_size,1,dim)
            task_input = torch.bmm(gate_i,expert_nets_out)#torch.matmul(gate_i,expert_nets_out)
            task_outs.append(task_net(task_input.squeeze()))
        return torch.cat(task_outs,dim=-1)

    def loss(self,Y):
        torch.tensor([1]).softmax()