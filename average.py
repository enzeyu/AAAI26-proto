import copy
import torch
from collections import defaultdict
from torch import nn

def average_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  #the nn layer loop
        for i in range(1, len(w)):   #the client loop
            w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num/total_sample_num)
    return w_avg

def agg_func(protos):
    """
    Returns the average of the weights.
    """
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos

def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label

def edge_proto_aggregation(edge_protos_all):
    # print(len(edge_protos_all))
    agg_protos_label = defaultdict(list)
    for edge_protos in edge_protos_all:
        for label in edge_protos.keys():
            agg_protos_label[label].append(edge_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data
    return agg_protos_label