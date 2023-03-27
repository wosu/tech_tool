import time
from typing import List

import torch
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.optim import Optimizer

from python.pytorch.recsys_models.dataset import load_ctr_datas
from python.pytorch.utils.metrics import precision, tensor_precision, tensor_recall
from python.pytorch.utils.model_util import model_visual, model_visual_by_hiddenlayer
from python.pytorch.utils.pytorch_util import try_gpu, tensor2numpy


class LR(nn.Module):
    def __init__(self,dis_feafield_size:list,feature_num,dis_fea_num):
        super(LR,self).__init__()
        # 每个离散特征域下对应特征大小
        # [x1,x2,x3..xm,x_m+1,...] 从x_m+1开始为连续特征
        self.fea_num = feature_num
        self.dis_fea_num = dis_fea_num
        self.conti_fea_num = feature_num-dis_fea_num
        # self.dis_fea_embeddings=[]这样创建的embedding表是存放在cpu上，要使用pytorch的组件创建
        # 这样才会存放在模型指定的device上
        self.dis_fea_embeddings = nn.ModuleList()
        for fea_size in dis_feafield_size:
            self.dis_fea_embeddings.append(nn.Embedding(fea_size,1))
        self.activation = nn.Sigmoid()
        # self.bias = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    def forward(self,X):
        # X shape:(batch_size,fea_num) 每个离散特征的特征为对应的index
        # 切片索引获取所有batch中的连续特征
        conti_feas = X[:,self.dis_fea_num:]
        sample_dis_feas = X[:,:self.dis_fea_num]

        batch_feas_embed = []
        # 遍历离散特征，得到batch中每个离散特征对应的embedding
        for i in range(self.dis_fea_num):
            # 获取第i个离散特征域的embedding table
            dis_fea_embedding_table = self.dis_fea_embeddings[i]
            # batch_fea_idxs shape:(1,batch_size)
            batch_fea_idxs = sample_dis_feas[:,i].long()
            # dis_fea_embedding_table(batch_fea_idxs) shape:(batch_size,embed_size)
            batch_feas_embed.append(dis_fea_embedding_table(batch_fea_idxs))
        # dis_line_part shape:(batch_size,dis_fea_num)
        dis_linear_part = torch.cat(batch_feas_embed,dim=1)
        linear_part = torch.cat([dis_linear_part,conti_feas],dim=1)
        # (batch_size,1)
        return self.activation(torch.sum(linear_part,dim=1,keepdim=True))


def train(model:nn.Module,train_iter,valid_iter,num_epochs,updater:Optimizer,device):
    print(f"running on device:{device}")
    lr_scheduler_step = 10
    if isinstance(lr_scheduler_step, int) and lr_scheduler_step > 0:
        lr_scheduler = optim.lr_scheduler.StepLR(updater, step_size=lr_scheduler_step, gamma=0.1)
    elif isinstance(lr_scheduler_step, List) and len(lr_scheduler_step) > 0:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(updater, milestones=lr_scheduler_step, gamma=0.1)
    else:
        lr_scheduler = None

    model:nn.Module = model.to(device)
    # model.apply(model.init_weights)
    print('model.dis_fea_embeddings[0].weight',model.dis_fea_embeddings[0].weight)


    loss = nn.BCELoss()
    for epoch in range(num_epochs):
        st = time.time()
        model.train()
        train_y_true = torch.empty(0,device=device)
        train_y_pred = torch.empty(0,device=device)
        print(f"start epoch:{epoch}")
        for X,Y in train_iter:
            # 设置训练模式
            # 将数据拷贝到GPU上进行训练
            X = X.to(device)
            Y = Y.to(device)
            Y_pred = model(X).to(device)
            # 计算loss
            ls = loss(Y_pred,Y)
            # 反向转播前，需要将累积梯度清零
            updater.zero_grad()
            # 误差反向传播
            ls.backward()
            # 优化器通过梯度下降对参数进行更新
            updater.step()

            train_y_true = torch.cat([train_y_true,Y],dim=0)
            train_y_pred = torch.cat([train_y_pred,Y_pred],dim=0)
        test_y_true = torch.empty(0,device=device)
        test_y_pred = torch.empty(0,device=device)
        # 每个epoch结束使用模型对验证集进行验证,将模型设置成验证模式
        model.eval()
        for X_test,Y_test in valid_iter:
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)
            # 不进行梯度计算
            with torch.no_grad():
                Y_pred_test = model(X_test).to(device)
                test_y_pred = torch.cat([test_y_pred,Y_pred_test],dim=0)
                test_y_true = torch.cat([test_y_true,Y_test],dim=0)

        # 计算评价指标
        train_precision = tensor_precision(train_y_true,train_y_pred)
        test_precision = tensor_precision(test_y_true,test_y_pred)

        train_recall = tensor_recall(train_y_true, train_y_pred)
        test_recall = tensor_recall(test_y_true, test_y_pred)

        train_auc = roc_auc_score(y_true=tensor2numpy(train_y_true), y_score=tensor2numpy(train_y_pred))
        valid_auc = roc_auc_score(y_true=tensor2numpy(test_y_true), y_score=tensor2numpy(test_y_pred))

        et = time.time()
        time_cost = et-st
        print(f"epoch {epoch}:time cost:{time_cost}second, train_precision:{train_precision} test_precision:{test_precision}, train_recall:{train_recall} test_recall:{test_recall},train_auc:{train_auc} test_auc:{valid_auc} ")
        #每次epoch结束后，更新学习率
        lr_scheduler.step()
    model = model.to("cpu")
    # 保存模型参数
    torch.save(model.state_dict(),"../model_result/lr_model.pt")
    # jit trace保存完整模型，可供线上通过pytorch jit trace加载模型使用
    #torch.jit.trace(model,X).save("../model_result/lr_model.trace")


if __name__ == "__main__":
    # dis_num = 2
    # dis_feafield_size = [5,4]
    # test_embed_table_list =  [nn.Embedding(fea_size,1) for fea_size in dis_feafield_size]
    # batch_data = torch.tensor([[1,2.0,3.4],[0,3,4.7],[1,0,3.4]],dtype=torch.float)
    # print(batch_data[:,:2].int())
    # batch_embed_data = ]
    # for i in range(dis_num):
    #     print(i)
    #     batch_fea_idx = batch_data[:,i].type(torch.int)
    #     print("batch_fea_idx:",batch_fea_idx.dtype,batch_fea_idx)
    #
    #     print(test_embed_table_list[i].weight)
    #     batch_embed_data.append(test_embed_table_list[i].forward(batch_fea_idx))
    #     print(test_embed_table_list[i].forward(batch_fea_idx))
    # print('batch_embed_data',batch_embed_data)
    # print(torch.cat(batch_embed_data,dim=1))
    # print(torch.stack(batch_embed_data,dim=0).shape,torch.stack(batch_embed_data,dim=1))
    dis_field_size, schema, dis_schema, train_loader, test_loader = load_ctr_datas(file_path="../datas/mobile@default", batch_size=4096, user_gpu=True)
    # schema = [1 for i in range(10)]
    # dis_schema = [1 for i in range(4)]
    # dis_field_size = [10,12,15,16]

    lr_model = LR(dis_feafield_size=dis_field_size,feature_num=len(schema),dis_fea_num=len(dis_schema))

    print(lr_model.state_dict())
    for p in lr_model.parameters():
        print("para:",p)
    num_epochs = 5
    device = try_gpu(0)
    updater = torch.optim.Adam(params=lr_model.parameters(), lr=0.01, weight_decay=0.0001)
    train(lr_model,train_loader,test_loader,num_epochs=num_epochs,updater=updater,device=device)
    lr_model.load_state_dict(torch.load("../model_result/lr_model.pt"))
    x = torch.tensor([[1.0 for i in range(len(schema))]])
    model_visual(lr_model,'lr_model_visual',x,"../model_result/")


