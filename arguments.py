import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_type', type=str, default='ad', choices=['oodd', 'ad'])
    parser.add_argument('-DS', help='Dataset', default='BZR')
    parser.add_argument('-DS_ood', help='Dataset', default='COX2')
    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    # 0.0001
    parser.add_argument('-lr', type=float, default=0.0001)
    # parser.add_argument('-num_layer', type=int, default=5)
    # parser.add_argument('-hidden_dim', type=int, default=16)
    parser.add_argument('-num_trial', type=int, default=5)
    parser.add_argument('-num_epoch', type=int, default=400)
    parser.add_argument('-eval_freq', type=int, default=1)
    parser.add_argument('-is_adaptive', type=int, default=1)
    # parser.add_argument('-num_cluster', type=int, default=2)
    parser.add_argument('-alpha', type=float, default=0)
    parser.add_argument('-hyper_c', type=float, default= 0.01,
                        help='balance between hyperbolic space and Euclidean space')
    parser.add_argument('-clip_r', type=float, default=2.3, help='feature clip radius')
    # parser.add_argument('-IC', action='store_true', help=' instance-wise contrastive learning without hierarchies')

    parser.add_argument('-HIC', action='store_true', help=' hierarchical instance-wise contrastive learning')
    parser.add_argument('-HPC', action='store_true', help=' prototypical contrastive learning')
    parser.add_argument('-cluster_num', default='5,1', type=str,
                        help='number of clusters')
    parser.add_argument('-tau', default=0.2, type=float,
                        help='softmax temperature')
    # parser.add_argument('-alpha_IB', type=float, default=0.5)
    parser.add_argument('-encoder_layers', type=int, default=4)
    parser.add_argument('-hidden_dim', type=int, default=16)
    parser.add_argument('-pooling', type=str, default='add', choices=['add', 'max'])
    parser.add_argument('-readout', type=str, default='concat', choices=['concat', 'add', 'last'])
    parser.add_argument('-type_learner', type=str, default='gnn', choices=["none", "att", "mlp", "gnn"])
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.4)
    parser.add_argument('-reg_lambda', type=float, default=10)
    return parser.parse_args()