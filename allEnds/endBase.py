from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from data.data_utils import read_end_data
from sklearn.preprocessing import label_binarize
from torch.optim import lr_scheduler
import numpy as np
from sklearn import metrics

class EndBase(object):
    def __init__(self, index, args, model):
        self.index = index
        self.train_loader = self.load_train_data(args.dataset, self.index, args.batch_size)
        self.test_loader = self.load_test_data(args.dataset, self.index, args.batch_size)
        self.model = model
        self.batch_size = args.batch_size
        self.train_samples = len(self.train_loader) * self.batch_size
        self.loss = nn.CrossEntropyLoss()
        self.local_epoch = args.num_local_training
        self.learning_rate = args.lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[20,40,60,80,100], gamma=0.9)
        self.parent = None  
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.receiver_knowledges = {}
        self.sender_knowledges = {}

    def receive_from_edge(self, global_knowledge):
        self.receiver_knowledges = global_knowledge
        return None
    
    def receive_model_from_edge(self):
        self.model = self.receiver_knowledges

    def send_to_edge(self):
        self.parent.receiver_knowledges[self.index] = self.sender_knowledges
        return None
    
    def load_train_data(self, dataset, index, batch_size=None):
        train_data = read_end_data(dataset, index, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
    
    def load_test_data(self, dataset, index, batch_size=None):
        test_data = read_end_data(dataset, index, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)


    def test_metrics(self):
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        
        # if np.isnan(y_prob).any() or np.isnan(y_true).any():
        #     print(y_prob, y_true)
        #     raise ValueError("y_prob or y_true contains NaN values")

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        return test_acc, test_num, auc

    def train_metrics(self):
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
