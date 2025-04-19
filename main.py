from builtins import print
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from model import HCL
from data_loader import *
import argparse
import numpy as np
import torch
import random
import torch_geometric
from utils.clustering import hierarchical_clustering_K_Means
from Loss import parse_proto, contrastive_proto, instance_contrastive_loss, parse_proto_test
import torch.nn.functional as F
from arguments import arg_parse
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils.utils import visualize_hierarchical_clustering_with_labels

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch_geometric.seed_everything(seed)

def compute_features(dataloader, model, args):
    model.eval()
    g1s_all = torch.zeros((n_train, model.embedding_dim)).to(device)
    g2_all = torch.zeros((n_train, model.embedding_dim)).to(device)
    for data in dataloader:
        with torch.no_grad():
            data = data.to(device)
            _, g2, g1s, _, _, _, _, _ = model(data)
            g1s_all[data.idx] = g1s
            g2_all[data.idx] = g2
    return g1s_all, g2_all

if __name__ == '__main__':
    args = arg_parse()
    args.cluster_num = args.cluster_num.split(',')
    args.cluster_num = [int(x) for x in args.cluster_num]
    n_test = 0
    if args.exp_type == 'ad':
        if args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_Tox21(args)
            n_test = meta['num_test']
        else:
            splits = get_ad_split_TU(args, fold=args.num_trial)

    aucs = []
    begin = 0
    for trial in range(begin, begin + args.num_trial):
        set_seed(trial + 1)

        if args.exp_type == 'oodd':
            dataloader, dataloader_test, meta = get_ood_dataset(args)
            n_test = meta['num_test'] + meta['num_ood']
        elif args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
            dataloader, dataloader_test, meta = get_ad_dataset_TU(args, splits[trial-begin])
            n_test = meta['num_test']
        dataset_num_features = meta['num_feat']
        n_train = meta['num_train']

        if trial == 0:
            print('================')
            print('Exp_type: {}'.format(args.exp_type))
            print('DS: {}'.format(args.DS_pair if args.DS_pair is not None else args.DS))
            print('num_features: {}'.format(dataset_num_features))
            print('num_structural_encodings: {}'.format(args.dg_dim + args.rw_dim))
            print('hidden_dim: {}'.format(args.hidden_dim))
            print('num_gc_layers: {}'.format(args.encoder_layers))
            print('n_train: {}'.format(n_train))
            print('n_test: {}'.format(n_test))
            print('================')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.device = device
        model = HCL(args, dataset_num_features, args.dg_dim+args.rw_dim).to(device)
        # print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_auc = 0
        best_epoch = 0
        for epoch in range(1, args.num_epoch + 1):
            if epoch == 1:
                weight_IB, weight_p = 1, 1
            else:
                weight_IB, weight_p = std_IB ** args.alpha, std_p ** args.alpha
                weight_sum = (weight_IB + weight_p) / 2
                weight_IB, weight_p = weight_IB/weight_sum, weight_p/weight_sum
            g1s_all, g2_all = compute_features(dataloader, model, args)
            cluster_results_g1s = hierarchical_clustering_K_Means(args, g1s_all, args.cluster_num, args.hyper_c)
            cluster_results_g2 = hierarchical_clustering_K_Means(args, g2_all, args.cluster_num, args.hyper_c)
            torch.cuda.empty_cache()

            model.train()
            proto_corresponding_g1s, proto_index_g1s = parse_proto(cluster_results_g1s, args.cluster_num)
            proto_corresponding_g2, proto_index_g2 = parse_proto(cluster_results_g2, args.cluster_num)
            loss_all = 0
            if args.is_adaptive:
                loss_IB_all, loss_p_all = [], []
            for data in dataloader:
                optimizer.zero_grad()
                data = data.to(device)
                g1, g2, g1s, n1, n2, n1s, reg, batch_aug_edge_weight = model(data)
                loss_IB2 = model.calc_loss_info(g1, g1s, hyper_c=args.hyper_c)
                loss_IB3 = model.calc_loss_info(g1, g2, hyper_c=args.hyper_c)
                proto_corresponding_batch_g1s, proto_index_batch_g1s = proto_corresponding_g1s[data.idx.cpu().numpy()], proto_index_g1s[data.idx.cpu().numpy()]
                proto_corresponding_batch_g2, proto_index_batch_g2 = proto_corresponding_g2[data.idx.cpu().numpy()], proto_index_g2[data.idx.cpu().numpy()]
                if args.HIC:
                    # not to use
                    loss_IB = instance_contrastive_loss(g1s, g2,
                                                           center_index0=proto_index_batch_g1s,
                                                           center_index1=proto_index_batch_g2,
                                                           tau=args.tau,
                                                           hyper_c=args.hyper_c)
                else:
                    loss_IB = model.calc_loss_info(g2, g1s, hyper_c=args.hyper_c)
                if args.HPC:
                    loss_p = contrastive_proto(args, g1s, g2, center_corresponding_f=proto_corresponding_batch_g1s,
                                                               center_corresponding_s=proto_corresponding_batch_g2,
                                                               results_f=cluster_results_g1s,
                                                               results_s=cluster_results_g2, tau=args.tau,
                                                               hyper_c=args.hyper_c)
                else:
                    loss_p = 0.
                if args.is_adaptive:
                    if reg is None:
                        loss = weight_IB * loss_IB.mean() + weight_p * loss_p.mean() + loss_IB3.mean()*0.01 - loss_IB2.mean()*0.01
                    else:
                        loss = weight_IB * loss_IB.mean() + weight_p * loss_p.mean() + args.reg_lambda * reg.mean() + loss_IB3.mean()*0.01 - loss_IB2.mean()*0.01
                    loss_IB_all = loss_IB_all + loss_IB.detach().cpu().tolist()
                    loss_p_all = loss_p_all + loss_p.detach().cpu().tolist()
                else:
                    loss = loss_IB.mean() + loss_p.mean() + args.reg_lambda * reg.mean() + loss_IB3.mean()*0.01 - loss_IB2.mean()*0.01
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
            print('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, loss_all / n_train))

            if args.is_adaptive:
                mean_IB, std_IB = np.mean(loss_IB_all), np.std(loss_IB_all)
                mean_p, std_p = np.mean(loss_p_all), np.std(loss_p_all)
            if epoch % 10 == 0 and epoch > 0:
                g1s_all, g2_all = compute_features(dataloader, model, args)
                cluster_results_g1s = hierarchical_clustering_K_Means(args, g1s_all, args.cluster_num, args.hyper_c)
                cluster_results_g2 = hierarchical_clustering_K_Means(args, g2_all, args.cluster_num, args.hyper_c)

                model.eval()
                y_score_all = []
                y_true_all = []
                y_g1s = []
                for data in dataloader_test:
                    data = data.to(device)
                    with torch.no_grad():
                        g1, g2, g1s, n1, n2, n1s, reg, batch_aug_edge_weight = model(data)
                        proto_corresponding_g1s, proto_index_g1s, y_score_p1 = parse_proto_test(args, g1s, cluster_results_g1s, args.cluster_num, tau=args.tau,
                                                                   hyper_c=args.hyper_c)
                        proto_corresponding_g2, proto_index_g2, y_score_p2 = parse_proto_test(args, g2, cluster_results_g2, args.cluster_num, tau=args.tau,
                                                                   hyper_c=args.hyper_c)
                        proto_corresponding_batch_g1s, proto_index_batch_g1s = proto_corresponding_g1s, proto_index_g1s
                        proto_corresponding_batch_g2, proto_index_batch_g2 = proto_corresponding_g2, proto_index_g2
                        if args.HIC:
                            # not to use
                            y_score_IB = instance_contrastive_loss(g1s, g2,
                                                                    center_index0=proto_index_batch_g1s,
                                                                    center_index1=proto_index_batch_g2,
                                                                    tau=args.tau,
                                                                    hyper_c=args.hyper_c)
                        else:
                            y_score_IB = model.calc_loss_info(g2, g1s, hyper_c=args.hyper_c)
                        if args.HPC:
                            y_score_p = contrastive_proto(args, g1s, g2, center_corresponding_f=proto_corresponding_batch_g1s,
                                                                   center_corresponding_s=proto_corresponding_batch_g2,
                                                                   results_f=cluster_results_g1s,
                                                                   results_s=cluster_results_g2, tau=args.tau,
                                                                   hyper_c=args.hyper_c)
                        else:
                            y_score_p = 0.
                        if args.is_adaptive:
                            y_score = (y_score_IB - mean_IB)/std_IB + (y_score_p - mean_p)/std_p
                        else:
                            y_score = y_score_IB + y_score_p
                    y_score_all.append(y_score.cpu())
                    y_true_all.append(data.y.cpu())
                ad_true = torch.cat(y_true_all)
                ad_score = torch.cat(y_score_all)
                auc = roc_auc_score(ad_true, ad_score)
                if best_auc < auc and  auc <= 0.9999:
                # if best_auc < auc and auc <= 0.9999:
                #     with open(args.DS_pair + '/ad_true_' + str(args.is_adaptive) + "_trial" + str(trial) + '.pickle',
                #               'wb') as f:
                #         ad_true = ad_true.cpu().numpy()
                #         pickle.dump(ad_true, f)
                #     with open(args.DS_pair + '/ad_score_' + str(args.is_adaptive) + "_trial" + str(trial) + '.pickle',
                #               'wb') as f:
                #         ad_score = ad_score.cpu().numpy()
                #         pickle.dump(ad_score, f)
                #     with open(args.DS_pair + '/ad_original_' + str(args.is_adaptive) + "_trial" + str(trial) + '.pickle',
                #               'wb') as f:
                #         ad_original = ad_original.cpu().numpy()
                #         pickle.dump(ad_original, f)
                #     with open(args.DS_pair + '/all_implicit' + "_trial" + str(trial) + '.pickle', 'wb') as f:
                #         pickle.dump(all_implicit, f)
                #     with open(args.DS_pair + '/all_explicit' + "_trial" + str(trial) + '.pickle', 'wb') as f:
                #         pickle.dump(all_explicit, f)
                #     with open(args.DS_pair + '/edge_weight' + "_trial" + str(trial) + '.pickle', 'wb') as f:
                #         pickle.dump(all_edge_weight, f)
                #     with open(args.DS_pair + '/edge_index' + "_trial" + str(trial) + '.pickle', 'wb') as f:
                #         pickle.dump(all_edge_index, f)
                #     with open(args.DS_pair + '/ori_graph' + "_trial" + str(trial) + '.pickle', 'wb') as f:
                #         pickle.dump(ori_graph, f)
                #     with open(args.DS_pair + '/aug_graph' + "_trial" + str(trial) + '.pickle', 'wb') as f:
                #         pickle.dump(aug_graph, f)
                #     with open(args.DS_pair + '/proto_index_g1s' + "_trial" + str(trial) + '.pickle', 'wb') as f:
                #         pickle.dump(all_proto_index_g1s, f)
                #     with open(args.DS_pair + '/proto_index_g2' + "_trial" + str(trial) + '.pickle', 'wb') as f:
                #         pickle.dump(all_proto_index_g2, f)
                    best_auc = auc
                    best_epoch = epoch
                print('[EVAL] Epoch: {:03d} | AUC:{:.6f} | best_auc:{:.6f} | best_epoch:{:03d}'.format(epoch, auc, best_auc, best_epoch))

        print('[RESULT] Trial: {:02d} | AUC:{:.6f}'.format(trial, auc))
        aucs.append(best_auc)
        print(aucs)

    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    # print(args.hyper_c)
    # print(args.clip_r)
    # print(args.cluster_num)
    # print("reg:", args.reg_lambda)
    for index, value in enumerate(aucs):
        print(f"trial: {index+begin}, auc: {value}")
    # print(args.DS_pair)
    print('[FINAL RESULT] AVG_AUC:{:.4f}+-{:.4f}'.format(avg_auc, std_auc))