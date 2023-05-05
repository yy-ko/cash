import random, os, sys
import numpy as np
import time, statistics
import logging, warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from sklearn import metrics
from torchmetrics import AveragePrecision

import models, data, utils
import pdb
from dgl import DGLGraph

warnings.simplefilter("ignore")


# for Reproducibility
def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def evaluate(model, g, n_feat, he_feat, dataloader, iters, method):
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for _ in range(iters):
            # 1. HNHN message passing
            dummy_mask = utils.gen_feature_mask(0)
            nfeat, efeat = model(g, dummy_mask, n_feat, he_feat)

            # 2. candidate scoring for hyperedge in validation/test datasets
            hedges, labels = dataloader.next() 
            test_preds += model.aggregate(nfeat, hedges, mode='Eval', method=method) 
            test_labels.append(labels.detach())

        test_preds = torch.sigmoid(torch.stack(test_preds).squeeze())
        test_labels = torch.cat(test_labels, dim=0)

    return test_preds.tolist(), test_labels.tolist()



def train(args, data_info, node_aggr_info, device):

    best_accuracy = [0.0 for _ in range(args.num_split)]
    best_epoch = [0 for _ in range(args.num_split)]

    for split in range(args.num_split): # number of splits (default: 5)
        data_dict = torch.load(f'./data/splits/{args.dataset}split{split}.pt')
        ground = data_dict["ground_train"] + data_dict["ground_valid"] # ground_train + train_only?
        g = utils.gen_DGLGraph_with_droprate(ground, 0, method=args.augment_method).to(device)

        # get dataloaders for training and validation datasets
        train_pos_loader, train_neg_loader = data.get_dataloaders(data_dict, args.batch_size, device, args.ns_method, label='Train')
        valid_pos_loader, valid_neg_sns_loader, valid_neg_mns_loader, valid_neg_cns_loader = data.get_dataloaders(data_dict, args.batch_size, device, None, label='Valid')
        train_iters, val_pos_iters, val_neg_iters = utils.get_num_iters(data_dict, args.batch_size, label='Train')

        # Initialize models
        # 1. Hypergraph encoder (shared in the two augmented views)
        n_feat, he_feat = data_info['node_feat'][g.nodes('node')], data_info['hyperedge_feat'][g.nodes('hedge')]
        n_norm, he_norm = data_info['node_normalized'][g.nodes('node')], data_info['edge_normalized'][g.nodes('hedge')]
        n_norm_sum, he_norm_sum = data_info['node_normalized_sum'][g.nodes('node')], data_info['edge_normalized_sum'][g.nodes('hedge')]

        encoder = models.HypergraphEncoder(args.h_dim, data_info['input_dim'], args.dropout, n_norm, n_norm_sum, he_norm, he_norm_sum)
        model = models.OurModel(encoder, args.proj_dim, node_aggr_info).to(device)

        # 2. Classifier for candidate scoring
        model_params = list(model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10) # learning rate scheduling

        average_precision = AveragePrecision()

        # Training phase
        # 1. Hypergraph encoder (HGNN model) + Projection
        # 2. Candidate scoring (positive + negative)
        # 3. Loss computation and backpropagation
        print(f'============================================ Split {split} ==================================================')
        print('#Epoch \t Train Loss \t ROC SNS | MNS | CNS | Mixed | Average \t AP SNS | MNS | CNS | Mixed | Average')

        patience_epoch = 0
        for epoch in range(args.num_epochs):

            model.train()
            total_loss = 0.0
            train_pred, train_label = [], []

            for _ in range(train_iters):
                # generating two augmented views for contrastive learning
                g1 = utils.gen_DGLGraph_with_droprate(ground, args.drop_incidence_rate, method=args.augment_method).to(device)
                g2 = utils.gen_DGLGraph_with_droprate(ground, args.drop_incidence_rate, method=args.augment_method).to(device)
                n_mask1 = utils.gen_feature_mask(args.drop_feature_rate)
                n_mask2 = utils.gen_feature_mask(args.drop_feature_rate)

                n_mask0 = utils.gen_feature_mask(0.0)

                # 1. Hypergraph Encoder 
                n, he = model(g, n_mask0, n_feat, he_feat)

                n1, he1 = model(g1, n_mask1, n_feat, he_feat)
                n2, he2 = model(g2, n_mask2, n_feat, he_feat)

                # 1-2. Projection
                np1, np2 = model.node_projection(n1), model.node_projection(n2)
                hep1, hep2 = model.hedge_projection(he1), model.hedge_projection(he2)

                # 2. candidate scoring for both positive and negative hyperedges
                pos_hedges, pos_labels = train_pos_loader.next()
                neg_hedges, neg_labels = train_neg_loader.next()

                pos_preds = model.aggregate(n, pos_hedges, mode='Train', method=args.aggre_method)
                neg_preds = model.aggregate(n, neg_hedges, mode='Train', method=args.aggre_method)
                #  neg_preds1, neg_preds2 = model.aggregate(n1, neg_hedges, mode='Train', method=args.aggre_method), model.aggregate(n2, neg_hedges, mode='Train', method=args.aggre_method)

                # 3. compute training loss and update parameters
                d_real_loss = bce_loss(pos_preds, pos_labels) 
                d_fake_loss = bce_loss(neg_preds, neg_labels) 
                #  d_real_loss = (bce_loss(pos_preds1, pos_labels) + bce_loss(pos_preds2, pos_labels)) / 2
                #  d_fake_loss = (bce_loss(neg_preds1, neg_labels) + bce_loss(neg_preds2, neg_labels)) / 2

                pred_loss = d_real_loss + d_fake_loss
                contrast_loss = -(torch.log(model.cosine_similarity(np1, np2)) + torch.log(model.cosine_similarity(hep1, hep2)))

                if args.use_contrastive == 1:
                    train_loss = pred_loss + (contrast_loss*args.contrast_ratio)
                else:
                    train_loss = pred_loss

                train_loss.backward()
                nn.utils.clip_grad_norm_(model_params, args.clip)
                optimizer.step()

                total_loss += train_loss.item()
            epoch_loss = total_loss / train_iters
            scheduler.step(epoch_loss)

            # Evaluation phase
            # 1. postiive dataset + four negative datasets (SNS, MNS, CNS, and Mixed)
            val_pred_pos, val_label_pos = evaluate(model, g, n_feat, he_feat, valid_pos_loader, val_pos_iters, args.aggre_method)
            val_pred_sns, val_label_sns = evaluate(model, g, n_feat, he_feat, valid_neg_sns_loader, val_neg_iters, args.aggre_method)
            val_pred_mns, val_label_mns = evaluate(model, g, n_feat, he_feat, valid_neg_mns_loader, val_neg_iters, args.aggre_method)
            val_pred_cns, val_label_cns = evaluate(model, g, n_feat, he_feat, valid_neg_cns_loader, val_neg_iters, args.aggre_method)

            # SNS validation set
            roc_sns = metrics.roc_auc_score(np.array(val_label_pos+val_label_sns), np.array(val_pred_pos+val_pred_sns))
            ap_sns = average_precision(torch.tensor(val_pred_pos+val_pred_sns), torch.tensor(val_label_pos+val_label_sns))

            # MNS validation set
            roc_mns = metrics.roc_auc_score(np.array(val_label_pos+val_label_mns), np.array(val_pred_pos+val_pred_mns))
            ap_mns = average_precision(torch.tensor(val_pred_pos+val_pred_mns), torch.tensor(val_label_pos+val_label_mns))

            # CNS validation set
            roc_cns = metrics.roc_auc_score(np.array(val_label_pos+val_label_cns), np.array(val_pred_pos+val_pred_cns))
            ap_cns = average_precision(torch.tensor(val_pred_pos+val_pred_cns), torch.tensor(val_label_pos+val_label_cns))

            # Mixed validation set
            d = len(val_pred_pos) // 3
            val_label_mixed = val_label_pos + val_label_sns[0:d]+val_label_mns[0:d]+val_label_cns[0:d]
            val_pred_mixed = val_pred_pos + val_pred_sns[0:d]+val_pred_mns[0:d]+val_pred_cns[0:d]
            roc_mixed = metrics.roc_auc_score(np.array(val_label_mixed), np.array(val_pred_mixed))
            ap_mixed = average_precision(torch.tensor(val_pred_mixed), torch.tensor(val_label_mixed))

            roc_average = (roc_sns+roc_mns+roc_cns+roc_mixed)/4
            ap_average = (ap_sns+ap_mns+ap_cns+ap_mixed)/4

            print(f' {epoch}: \t {epoch_loss:.4f} \t {roc_sns:.4f} {roc_mns:.4f} {roc_cns:.4f} {roc_mixed:.4f} {roc_average:.4f} \t {ap_sns:.4f} {ap_mns:.4f} {ap_cns:.4f} {ap_mixed:.4f} {ap_average:.4f}')

            if roc_average > best_accuracy[split]:
                best_accuracy[split] = roc_average
                best_epoch[split] = epoch
                patience_epoch = 0

                # save model here
                torch.save(model.state_dict(), f"{args.model_dir}/model_gpu{args.gpu_index}_{split}.pkt")
            else:
                patience_epoch += 1
                if patience_epoch >= 20:
                    print('=== Early Stopping')
                    break

        print(' ')
        print(f'=====\t Split: {split} \t Best Accuracy: {best_accuracy[split]:.4f} \t Best Epoch: {best_epoch[split]} \t=====')
        print(' ')

def test(args, data_info, node_aggr_info, device):

    sns_avg_roc = []
    sns_avg_ap = []
    mns_avg_roc = []
    mns_avg_ap = []
    cns_avg_roc = []
    cns_avg_ap = []
    mixed_avg_roc = []
    mixed_avg_ap = []
    average_avg_roc = []
    average_avg_ap = []


    print(' ')
    print('=========================================== Test Start ================================================')
    print('#Split \t ROC SNS | MNS | CNS | Mixed | Average \t AP SNS | MNS | CNS | Mixed | Average')
    for split in range(args.num_split): # number of splits (default: 5)
        data_dict = torch.load(f'./data/splits/{args.dataset}split{split}.pt')
        ground = data_dict["ground_train"] + data_dict["ground_valid"]
        g = utils.gen_DGLGraph_with_droprate(ground, 0).to(device)

        # get dataloaders for training and validation datasets
        test_pos_loader, test_neg_sns_loader, test_neg_mns_loader, test_neg_cns_loader = data.get_dataloaders(data_dict, args.batch_size, device, None, label='Test')
        test_pos_iters, test_neg_iters = utils.get_num_iters(data_dict, args.batch_size, label='Test')

        # Initialize models
        # 1. Node embedding model
        n_feat, he_feat = data_info['node_feat'][g.nodes('node')], data_info['hyperedge_feat'][g.nodes('hedge')]
        n_norm, he_norm = data_info['node_normalized'][g.nodes('node')], data_info['edge_normalized'][g.nodes('hedge')]
        n_norm_sum, he_norm_sum = data_info['node_normalized_sum'][g.nodes('node')], data_info['edge_normalized_sum'][g.nodes('hedge')]

        encoder = models.HypergraphEncoder(args.h_dim, data_info['input_dim'], args.dropout, n_norm, n_norm_sum, he_norm, he_norm_sum)
        model = models.OurModel(encoder, args.proj_dim, node_aggr_info).to(device)
        model.load_state_dict(torch.load(f"{args.model_dir}/model_gpu{args.gpu_index}_{split}.pkt"))

        average_precision = AveragePrecision()

        # Test phase
        # 1. postiive dataset + four negative datasets (SNS, MNS, CNS, and Mixed)
        test_pred_pos, test_label_pos = evaluate(model, g, n_feat, he_feat, test_pos_loader, test_pos_iters, args.aggre_method)
        test_pred_sns, test_label_sns = evaluate(model, g, n_feat, he_feat, test_neg_sns_loader, test_neg_iters, args.aggre_method)
        test_pred_mns, test_label_mns = evaluate(model, g, n_feat, he_feat, test_neg_mns_loader, test_neg_iters, args.aggre_method)
        test_pred_cns, test_label_cns = evaluate(model, g, n_feat, he_feat, test_neg_cns_loader, test_neg_iters, args.aggre_method)

        # SNS 
        roc_sns = metrics.roc_auc_score(np.array(test_label_pos+test_label_sns), np.array(test_pred_pos+test_pred_sns))
        ap_sns = average_precision(torch.tensor(test_pred_pos+test_pred_sns), torch.tensor(test_label_pos+test_label_sns)).numpy()
        sns_avg_roc.append(roc_sns)
        sns_avg_ap.append(ap_sns)

        # MNS 
        roc_mns = metrics.roc_auc_score(np.array(test_label_pos+test_label_mns), np.array(test_pred_pos+test_pred_mns))
        ap_mns = average_precision(torch.tensor(test_pred_pos+test_pred_mns), torch.tensor(test_label_pos+test_label_mns)).numpy()
        mns_avg_roc.append(roc_mns)
        mns_avg_ap.append(ap_mns)

        # CNS 
        roc_cns = metrics.roc_auc_score(np.array(test_label_pos+test_label_cns), np.array(test_pred_pos+test_pred_cns))
        ap_cns = average_precision(torch.tensor(test_pred_pos+test_pred_cns), torch.tensor(test_label_pos+test_label_cns)).numpy()
        cns_avg_roc.append(roc_cns)
        cns_avg_ap.append(ap_cns)

        # Mixed 
        d = len(test_pred_pos) // 3
        test_label_mixed = test_label_pos + test_label_sns[0:d]+test_label_mns[0:d]+test_label_cns[0:d]
        test_pred_mixed = test_pred_pos + test_pred_sns[0:d]+test_pred_mns[0:d]+test_pred_cns[0:d]
        roc_mixed = metrics.roc_auc_score(np.array(test_label_mixed), np.array(test_pred_mixed))
        ap_mixed = average_precision(torch.tensor(test_pred_mixed), torch.tensor(test_label_mixed)).numpy()
        mixed_avg_roc.append(roc_mixed)
        mixed_avg_ap.append(ap_mixed)

        roc_average = (roc_sns+roc_mns+roc_cns+roc_mixed)/4
        ap_average = (ap_sns+ap_mns+ap_cns+ap_mixed)/4
        average_avg_roc.append(roc_average)
        average_avg_ap.append(ap_average)

        print(f'{split} \t {roc_sns:.4f} {roc_mns:.4f} {roc_cns:.4f} {roc_mixed:.4f} {roc_average:.4f} \t {ap_sns:.4f} {ap_mns:.4f} {ap_cns:.4f} {ap_mixed:.4f} {ap_average:.4f}')

    final_sns_roc = sum(sns_avg_roc)/len(sns_avg_roc)
    final_mns_roc = sum(mns_avg_roc)/len(mns_avg_roc)
    final_cns_roc = sum(cns_avg_roc)/len(cns_avg_roc)
    final_mixed_roc = sum(mixed_avg_roc)/len(mixed_avg_roc)
    final_average_roc = sum(average_avg_roc)/len(average_avg_roc)

    final_sns_ap = sum(sns_avg_ap)/len(sns_avg_ap)
    final_mns_ap = sum(mns_avg_ap)/len(mns_avg_ap)
    final_cns_ap = sum(cns_avg_ap)/len(cns_avg_ap)
    final_mixed_ap = sum(mixed_avg_ap)/len(mixed_avg_ap)
    final_average_ap = sum(average_avg_ap)/len(average_avg_ap)

    if args.num_split > 1:
        std_sns_roc = statistics.stdev(sns_avg_roc)
        std_mns_roc = statistics.stdev(mns_avg_roc)
        std_cns_roc = statistics.stdev(cns_avg_roc)
        std_mixed_roc = statistics.stdev(mixed_avg_roc)
        std_average_roc = statistics.stdev(average_avg_roc)

        std_sns_ap = np.std(sns_avg_ap)
        std_mns_ap = np.std(mns_avg_ap)
        std_cns_ap = np.std(cns_avg_ap)
        std_mixed_ap = np.std(mixed_avg_ap)
        std_average_ap = np.std(average_avg_ap)
    else:
        std_sns_roc = 0.0 
        std_mns_roc = 0.0 
        std_cns_roc = 0.0 
        std_mixed_roc = 0.0 
        std_average_roc = 0.0 

        std_sns_ap = 0.0 
        std_mns_ap = 0.0 
        std_cns_ap = 0.0 
        std_mixed_ap = 0.0 
        std_average_ap = 0.0 


    print('============================================ Test End =================================================')
    print(' ')
    print('AUROC \t\t\t\t\t AP')
    print('SNS\tMNS\tCNS\tMixed\tAverage\tSNS\tMNS\tCNS\tMixed\tAverage')
    print(f'{final_sns_roc:.4f}\t{final_mns_roc:.4f}\t{final_cns_roc:.4f}\t{final_mixed_roc:.4f}\t{final_average_roc:.4f}\t{final_sns_ap:.4f}\t{final_mns_ap:.4f}\t{final_cns_ap:.4f}\t{final_mixed_ap:.4f}\t{final_average_ap:.4f}')
    print(f'{std_sns_roc:.4f}\t{std_mns_roc:.4f}\t{std_cns_roc:.4f}\t{std_mixed_roc:.4f}\t{std_average_roc:.4f}\t{std_sns_ap:.4f}\t{std_mns_ap:.4f}\t{std_cns_ap:.4f}\t{std_mixed_ap:.4f}\t{std_average_ap:.4f}')




if __name__ == '__main__':
    args = utils.parse_args()
    utils.print_summary(args)
    set_random_seeds(args.seed)

    device = torch.device("cuda:{}".format(args.gpu_index))
    data_info = data.get_datainfo(args, device)
    node_aggr_info = {'nhead': args.num_heads, 'nlayer': args.num_layers, 'h_dim': args.h_dim, 'dropout': args.dropout}

    train(args, data_info, node_aggr_info, device)
    test(args, data_info, node_aggr_info, device)



