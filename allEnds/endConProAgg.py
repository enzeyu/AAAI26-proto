import torch.nn as nn
import time, copy,torch
from collections import defaultdict
import torch.nn.functional as F
import numpy as np

from average import agg_func
from allEnds.endBase import EndBase
from allModel.model import FedAvgCNN, BaseHeadSplit

class EndConProAgg(EndBase):
    def __init__(self, index, args, model):             
        super().__init__(index, args, model)
        self.local_protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()
        self.device = args.device
        self.lamda = 2         
        self.alpha = 1         
        self.num_classes = args.num_classes

        if hasattr(self.model,'fc'):
            fc_backup = copy.deepcopy(self.model.fc)
            self.model.fc = nn.Identity()
            self.model = BaseHeadSplit(self.model, fc_backup)
    
    def train(self):
        start_time = time.time()
        
        self.model.train()
        local_protos = defaultdict(list)

        
        for epoch in range(self.local_epoch):
            for i, (img, label) in enumerate(self.train_loader):
                img = img.to(self.device)
                label = label.to(self.device)
                
                rep = self.model.base(img)
                
                output = self.model.head(rep)   
                
                loss = self.loss(output, label)

               
                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())             
                    for i, y in enumerate(label):                       
                        y_c = y.item()
                        if type(self.global_protos[y_c]) != type([]):   
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda  
                    
                if self.local_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())             
                    copy_local_protos = copy.deepcopy(self.local_protos)
                    different_distance = 0
                    same_distance = 0                                   
                    for i, yy in enumerate(label):                       
                        y_c = yy.item()                                  
                        same_distance += self.cos_distance(proto_new[i,:], copy_local_protos[y_c])
                        for _, c in enumerate(copy_local_protos):       
                            if y_c != c:                                
                                different_distance += self.cos_distance(proto_new[i,:], copy_local_protos[c])
                    contra_loss = -torch.log(same_distance / (same_distance + different_distance))
                    loss += self.alpha * contra_loss
                            
                
                for i, y in enumerate(label):
                    y_c = y.item()
                    local_protos[y_c].append(rep[i, :].detach().data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
        self.local_protos = agg_func(local_protos)
        self.send_protos(self.local_protos)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # def test_metrics(self):
    #     self.model.eval()

    #     test_acc = 0
    #     test_num = 0
    #     y_prob = []
    #     y_true = []
        
    #     with torch.no_grad():
    #         for x, y in self.test_loader:
    #             x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = self.model(x)
    #             
    #             test_num += y.shape[0]
    #           
    #             probabilities = F.softmax(output, dim=1)

    #             test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                
    #             
    #             rep = self.model.base(x)
    #             output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
    #             
    #             for i, r in enumerate(rep):
    #                
    #                 for j, pro in self.global_protos.items():
    #                     if type(pro) != type([]):
    #                         output[i, j] = self.loss_mse(r, pro)

    #             test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()        
    #     return test_acc, test_num, 0


    def test_metrics(self):
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        if self.global_protos is not None:
            with torch.no_grad():
                for x, y in self.test_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                   
                    rep = self.model.base(x)
                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        
                        for j, pro in self.global_protos.items():
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r, pro)

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    def train_metrics(self):
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                label = y
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)                             

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda  
                
              
                if self.local_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())             
                    copy_local_protos = copy.deepcopy(self.local_protos)
                    different_distance = 0
                    same_distance = 0                                   
                    for i, yy in enumerate(label):                       
                        y_c = yy.item()                                  
                        same_distance += self.cos_distance(proto_new[i,:], copy_local_protos[y_c])
                        for _, c in enumerate(copy_local_protos):       
                            if y_c != c:                                
                                different_distance += self.cos_distance(proto_new[i,:], copy_local_protos[c])
                    contra_loss = -torch.log(same_distance / (same_distance + different_distance))
                    loss += self.alpha * contra_loss 
                
                # print(y, y.shape)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num

    
    def receive_protos(self, global_protos):
        self.receiver_knowledges = global_protos
        self.global_protos = self.receiver_knowledges
    
    def send_protos(self, local_protos):
        self.sender_knowledges = local_protos

    def cos_distance(self, vec1, vec2):
        if len(vec2) == 0:
            return 0.0
            
        dot_product = torch.dot(vec1, vec2)
        # print("dot_product: ", dot_product)
        norm1 = torch.norm(vec1, p=2)
        norm2 = torch.norm(vec2, p=2)
    
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = torch.exp(dot_product / (norm1 * norm2))
        # print("dot_product: ", similarity)
        return similarity
        