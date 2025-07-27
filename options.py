import argparse

def args_parser(parser):
    parser.add_argument('-data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('-dataset', type = str, default = 'cifar10', help= 'name of the dataset: mnist, cifar10, Imagenet')
    parser.add_argument('-num_classes', "--num_classes", type=int, default=10, help= 'default class number for cifar10')
    parser.add_argument('-end_model', type = str, default = 'cnn', help='type of model')
    parser.add_argument('-edge_model', type = str, default = 'resnet10', help='type of model')
    parser.add_argument('-batch_size', type = int, default = 10, help= 'batch size when trained on ends and edges')
    parser.add_argument('-num_comm', type = int, default=100, help= 'number of communication rounds between ends and edges')
    parser.add_argument('-num_loc_tra','--num_local_training', type=int, default=1, help='number of local training')
    parser.add_argument('-lr', type = float, default = 0.005, help = 'learning rate of the SGD when trained on client')
    parser.add_argument('-device', "--device", type=str, default="cuda", choices=["cpu", "cuda"], help = 'device for training')
    parser.add_argument('-alg', "--algorithm", type=str, default="FedProAgg")
    parser.add_argument('-num_runtime', type=int, default=1)
    parser.add_argument('-eval_term', type=int, default=1)

    parser.add_argument('-frac', type = float, default = 1, help = 'fraction of participated ends in every edge')           # 每个边缘客户端的参与比例
    parser.add_argument('-num_ends', type = int, default = 20, help = 'number of all available clients')
    parser.add_argument('-num_edges', type = int, default= 4, help= 'number of edges')
    parser.add_argument('-seed', type = int, default = 0, help = 'random seed (defaul: 0)')
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)

    parser.add_argument("--T_agg", nargs="*", type=float, default=3.0, help="T_agg")
    args = parser.parse_args()
    return args


