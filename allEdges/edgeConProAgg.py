import torch.nn as nn
import time, copy, torch
import numpy as np

from allEdges.edgeBase import EdgeBase
from average import proto_aggregation, edge_proto_aggregation

class EdgeConProAgg(EdgeBase):
    def __init__(self, index, dids, args, model):
        super().__init__(index, dids, args, model)

        self.eval_term = args.eval_term
        self.num_classes = args.num_classes
        self.aggregated_edge = False    
        self.global_protos = []
        self.uploaded_ids = []
        self.uploaded_protos = []

        self.rs_test_acc = []
        self.rs_train_loss = []


    def train(self, index, f):
        s_time = time.time()
        if  index % self.eval_term == 0:
            edge_index, train_loss_list, test_acc_list, std_accs_list = self.all_evaluate(index)
            f.write("communication " + str(index) +" :\n")
            f.write("Edge " + str(self.index) +" :\n")
            f.write("train_loss "+str(train_loss_list)+ "\n")
            f.write("test_acc " + str(test_acc_list) +"\n")
            f.write("std_accs " + str(std_accs_list) + "\n")
            f.write("\n")

        for end in self.ends_registration:
            end.train()
        self.receive_protos()
        # print(self.uploaded_protos)
        self.edge_protos = proto_aggregation(self.uploaded_protos)

        if self.aggregated_edge == True:
            
            # if index > 0 and index % self.eval_term == 0:
            #     edge_index, train_loss_list, test_acc_list, std_accs_list = self.all_evaluate(index)
            #     f.write("communication " + str(index) +" :\n")
            #     f.write("train_loss "+str(train_loss_list)+ "\n")
            #     f.write("test_acc " + str(test_acc_list) +"\n")
            #     f.write("std_accs " + str(std_accs_list) + "\n")
            #     f.write("\n")

            self.edge_protos_all = [copy.deepcopy(self.edge_protos)]
            for neigh in self.neigh_registration:
                self.edge_protos_all.append(neigh.edge_protos)
            self.global_protos = edge_proto_aggregation(self.edge_protos_all)
            for neigh in self.neigh_registration:
                neigh.global_protos = self.global_protos
                print(f"Communication round %d, Normal Edge %d has received global protos from edge %d." % (index, neigh.index, self.index))
            if self.global_protos is not None:
                for neigh in self.neigh_registration:
                    for end in neigh.ends_registration:
                        end.receive_protos(self.global_protos)
                    print(f"Communication round %d, Normal Edge %d has sent global protos to its devices" % (index, neigh.index))
                for end in self.ends_registration:
                    end.receive_protos(self.global_protos)
                print(f"Communication round %d, Aggregtaion Edge %d has sent global protos to its devices" % (index, self.index))

        if index % 10 == 0 and len(self.global_protos)>0:
            pass


    def receive_protos(self):
        for client in self.ends_registration:
            self.uploaded_ids.append(client.index)
            self.uploaded_protos.append(client.sender_knowledges)

