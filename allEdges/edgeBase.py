from average import average_weights
import torch, copy
import numpy as np

class EdgeBase(object):
    def __init__(self, index, dids, args, model):
        self.index = index  
        self.dids = dids    
        self.ends_registration = []  
        self.model = model  
        self.end_global_model = None  
        self.batch_size = args.batch_size
        self.local_epoch = args.num_local_training
        self.learning_rate = args.lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.neigh_registration = []  
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.generated_model = None     
        self.number_samples_edges = {}  
        self.number_samples_ends = 0 
        self.receiver_knowledges = {}
        self.sender_knowledges = None
        self.sender_models = {}
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.global_model = copy.deepcopy(args.end_model)       

        self.uploaded_edge_ids = []
        self.uploaded_edge_weights = []
        self.uploaded_edge_models = []

    def end_register(self, end):
        self.ends_registration.append(end)
    
    # def receive_from_other_edges(self, edge_id, shared_state_dict):
    #     self.receiver_models[edge_id] = shared_state_dict
    #     return None
    
    # def aggregate_model(self):
    #     received_dict = [dict for dict in self.receiver_models.values()]
    #     for edge in range(self.neigh_registration):
    #         self.number_samples_edges[edge.index] = sum(self.samples_end.values())
    #     self.model = average_weights(w = received_dict, s_num = self.number_samples_edges)
    
    def send_to_other_edges(self, model):
        for param, new_param in zip(self.model.parameters(), model.parameters()):
            param.data = new_param.data.clone()

    def send_to_ends(self):
        for end in self.ends_registration:
            end.receive_from_edge(self.sender_knowledges)

    def send_to_ends_model(self):
        for end in self.ends_registration:
            global_state_dict = self.sender_knowledges.state_dict()
            end.model.load_state_dict(global_state_dict) 

    def send_models(self):
        for end in self.ends_registration:
            end.set_parameters(self.global_model)

    def receive_from_ends(self, end):
        for end in range(self.ends_registration):
            end.send_to_edge()

    def receive_models(self):
        tot_samples = 0
        for end in self.ends_registration:
            tot_samples += end.train_samples
            self.uploaded_ids.append(end.index)
            self.uploaded_weights.append(end.train_samples)
            self.uploaded_models.append(end.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def receive_models_from_edges(self):
        tot_samples = 0
        self.uploaded_edge_ids = []
        self.uploaded_edge_weights = []
        self.uploaded_edge_models = []
        tot_samples += self.number_samples_ends
        self.uploaded_edge_ids.append(self.index)
        self.uploaded_edge_weights.append(self.number_samples_ends)
        self.uploaded_edge_models.append(self.global_model)
        for edge in self.neigh_registration:
            tot_samples += edge.number_samples_ends
            self.uploaded_edge_ids.append(edge.index)
            self.uploaded_edge_weights.append(edge.number_samples_ends)
            self.uploaded_edge_models.append(edge.global_model)
        for i, w in enumerate(self.uploaded_edge_weights):
            self.uploaded_edge_weights[i] = w / tot_samples
    
    def aggregate_parameters(self):
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def aggregate_parameters_edge(self):
        # self.end_global_model = average_weights(self.uploaded_edge_models, self.uploaded_edge_weights)
        self.end_global_model = copy.deepcopy(self.uploaded_edge_models[0])
        for param in self.end_global_model.parameters():
            param.data.zero_()
        for w, client_model in zip(self.uploaded_edge_weights, self.uploaded_edge_models):
            self.add_parameters_edge(w, client_model)
    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w
    def add_parameters_edge(self, w, client_model):
        for server_param, client_param in zip(self.end_global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def test_metrics(self):        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.ends_registration:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            tot_auc.append(auc*ns)

        ids = [c.index for c in self.ends_registration]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):        
        num_samples = []
        losses = []
        for c in self.ends_registration:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.index for c in self.ends_registration]

        return ids, num_samples, losses
    

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        return train_loss, test_acc, np.std(accs)

    def all_evaluate(self,index):
        print(f"\nEvaluate ends models in edge %d" % self.index)
        
        indexs = []
        train_loss_list = []
        test_acc_list = []
        std_accs_list = []

        train_loss, test_acc, std_accs = self.evaluate()
        indexs.append(self.index)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        std_accs_list.append(std_accs)

        # for neigh in self.neigh_registration:
        #     print(f"\nEvaluate ends models in Edge %d" % neigh.index)
        #     train_loss, test_acc, std_accs = neigh.evaluate()
        #     indexs.append(self.index)
        #     train_loss_list.append(train_loss)
        #     test_acc_list.append(test_acc)
        #     std_accs_list.append(std_accs)
        print(f"\n-----------------------------------------------")
        return indexs, train_loss_list, test_acc_list, std_accs_list