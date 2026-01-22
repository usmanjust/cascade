import os
import time
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast, GradScaler

from dataset import LGCNDataset, RecDataset
from models.cascade_model import CASCADERec
import utils

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)
torch.cuda.manual_seed_all(2022)
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

parser = argparse.ArgumentParser(description="CASCADE: Next-Basket Recommendation")
parser.add_argument('--dataset', type=str, default='beauty',
                    help="Dataset: beauty | grocery | tafeng | sports")
parser.add_argument('--mode', type=str, default='test',
                    help="Mode: train | test | case")
parser.add_argument('--model', type=str, default='cascade',
                    help="Model: cascade")
parser.add_argument('--use_adaptive', action='store_true',
                    help="Enable Adaptive Contrastive Signal Learning")
parser.add_argument('--use_causal', action='store_true',
                    help="Enable Causal Dependency Module")
parser.add_argument('--epoch', type=int, default=-1, 
                    help='Specific epoch to test (-1 for best)')
args = parser.parse_args()

config = {}
config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dataset = args.dataset.lower().strip()
config['run_type'] = args.mode
config['model_name'] = args.model.lower().strip()

config['train_path'] = f"./data/{dataset}/train.txt"
config['val_path'] = f"./data/{dataset}/val.txt"
config['test_path'] = f"./data/{dataset}/test.txt"
config['case_path'] = f"./data/{dataset}/case.txt"

config['batch_size'] = 1024
config['lgc_latent_dim'] = 32
config['gru_latent_dim'] = 32
config['lightGCN_n_layers'] = 3
config['gru_num_layers'] = 2
config['layer_norm_eps'] = 1e-5
config['dropout'] = 0.5

config['epoch'] = 400
config['lr'] = 0.001
config['weight_decay'] = 10e-5
config['topk'] = [10, 30, 50]
config['val_step'] = 1
config['patience'] = 10

config['num_aug'] = 1
config['n_interest'] = 1

config['use_adaptive'] = bool(args.use_adaptive or config['model_name'] == 'cascade')
config['use_causal'] = bool(args.use_causal or config['model_name'] == 'cascade')

config['lambda_rec'] = 1.0
config['lambda_align'] = 1.0
config['lambda_adaptive'] = 0.15
config['lambda_causal'] = 0.3
config['lambda_polar'] = 0.8
config['gumbel_tau'] = 1.0
config['gumbel_tau_min'] = 0.7
config['gumbel_tau_decay'] = 0.998
config['infoNCE_temp'] = 0.07
config['causal_blend'] = 0.5

DL_KWARGS = dict(num_workers=4, pin_memory=True)

def collate_fn(data):
    bseq, bseq_len, btar = [], [], []
    for i in data:
        bseq.append(i[0])
        bseq_len.append(i[1])
        btar.append(i[2])
    bseq = pad_sequence(bseq, batch_first=True)
    bseq_len = torch.tensor(bseq_len)
    btar = torch.stack(btar, dim=0)
    return bseq, bseq_len, btar

def recloss(logits, targets):
    return -torch.sum(targets * torch.log(logits + 1e-9)) / torch.sum(targets) \
           -torch.sum((1 - targets) * torch.log(1 - logits + 1e-9)) / torch.sum(1 - targets)

def cllossnp(batch_pos_bseq_emb, batch_neg_bseq_emb):
    ce = nn.CrossEntropyLoss()
    sim_pp = torch.matmul(batch_pos_bseq_emb, batch_pos_bseq_emb.T)
    sim_nn = torch.matmul(batch_neg_bseq_emb, batch_neg_bseq_emb.T)
    sim_pn = torch.matmul(batch_pos_bseq_emb, batch_neg_bseq_emb.T)
    d = sim_pn.shape[-1]
    sim_pp[..., range(d), range(d)] = 0.0
    sim_nn[..., range(d), range(d)] = 0.0
    raw_scores1 = torch.cat([sim_pn, sim_pp], dim=-1)
    raw_scores2 = torch.cat([sim_nn, sim_pn.transpose(-1, -2)], dim=-1)
    all_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
    labels = torch.arange(2 * d, dtype=torch.long, device=all_scores.device)
    cl_loss = ce(all_scores, labels)
    return cl_loss

def logitsloss(batch_bseq_emb, batch_iseq_emb):
    return torch.mean(torch.pow(batch_bseq_emb - batch_iseq_emb, 2))

@torch.no_grad()
def evaluate_model(model, dataset_path, lgcn_dataset, config):
    model.eval()
    rec_dataset = RecDataset(dataset_path, lgcn_dataset.basket2id_dict, lgcn_dataset.item2id_dict)
    test_loader = DataLoader(rec_dataset,
                             batch_size=config['batch_size'],
                             collate_fn=collate_fn,
                             shuffle=False,
                             drop_last=False,
                             **DL_KWARGS)

    f1s = {k: 0.0 for k in config['topk']}
    hits = {k: 0.0 for k in config['topk']}
    gts = 0
    sample_num = 0

    for _, (bseq, bseq_len, btar) in enumerate(test_loader):
        sample_num += len(bseq_len)
        gts += len(torch.nonzero(btar))
        bseq = bseq.to(config['device'])
        bseq_len = bseq_len.to(config['device'])
        btar = btar.to(config['device'])

        out = model(bseq, bseq_len, 'test')
        logits = out['logits'] if isinstance(out, dict) else out
        T = 0.7
        logits = logits / T

        for k in config['topk']:
            top_idx = torch.topk(logits, dim=1, k=k).indices
            preds = torch.zeros_like(logits).scatter(1, top_idx, 1.0)

            if k == 30:
                f1s[k] += sum(utils.f1(preds, btar))
                hits[k] += utils.hit(preds, btar)
            else:
                f1s[k] += sum(utils.f1(preds, btar))
                hits[k] += utils.hit(preds, btar)

    out = {
        'F1@10': f1s[10] / sample_num,
        'F1@30': f1s[30] / sample_num,
        'F1@50': f1s[50] / sample_num,
        'HR@10': hits[10] / gts,
        'HR@30': hits[30] / gts,
        'HR@50': hits[50] / gts
    }
    return out

def build_model(config, lgcn_dataset):
    return CASCADERec(config, lgcn_dataset).to(config['device'])

def train_one_dataset(config, model_save_path, val_log_path):
    lgcn_dataset = LGCNDataset(config['train_path'], config['val_path'], config['test_path'])
    model = build_model(config, lgcn_dataset)

    best_val = 0.0
    count = 0
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if "adaptive_head" not in n], 'lr': config['lr']},
        {'params': [p for n, p in model.named_parameters() if "adaptive_head" in n], 'lr': config['lr'] * 5}
    ]
    optimizer = torch.optim.Adam(param_groups, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    scaler = GradScaler(device='cuda' if 'cuda' in config['device'] else 'cpu')

    train_dataset = RecDataset(config['train_path'], lgcn_dataset.basket2id_dict, lgcn_dataset.item2id_dict)
    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              collate_fn=collate_fn,
                              drop_last=True,
                              shuffle=True,
                              **DL_KWARGS)

    print("Training started...")

    for e in range(config['epoch']):
        start_time = time.time()
        model.train()

        print(f"\nEpoch {e+1}/{config['epoch']} | {datetime.now().strftime('%H:%M:%S')}")

        model.g_droped = None
        model.pos_g_droped = None
        model.neg_g_droped = None

        if e > 0:
            if config.get('use_adaptive', False) and hasattr(model, "adaptive_head"):
                with torch.no_grad():
                    edge_probs = model.adaptive_head(
                        model.embedding_basket.weight,
                        model.embedding_item.weight
                    )
                    lgcn_dataset.set_adaptive_edge_weights(edge_probs)
                    lgcn_dataset.clear_cache()
            
            if config.get('use_causal', False) and hasattr(model, "causal_layer"):
                with torch.no_grad():
                    causal_matrix = model.causal_layer.estimate_causal_matrix(
                        model.embedding_item.weight,
                        model.embedding_basket.weight
                    )
                    lgcn_dataset.set_causal_matrix(causal_matrix)

        if hasattr(model, "get_graph"):
            model.get_graph(run_type='aug')
        if hasattr(model, "augment"):
            model.augment()
        if hasattr(model, "get_graph"):
            model.get_graph(run_type='train')

        if config.get('use_adaptive', False) and hasattr(model, "set_gumbel_tau"):
            initial_tau = 1.0
            min_tau = 0.7
            decay_rate = 0.99
            tau = max(min_tau, initial_tau * (decay_rate ** e))
            model.set_gumbel_tau(tau)

        total_rec, total_cl, total_lgt = 0.0, 0.0, 0.0
        total_adp, total_cau, total_pol = 0.0, 0.0, 0.0
        batch_count = 0

        for batch_idx, (bseq, bseq_len, btar) in enumerate(train_loader, 1):
            bseq = bseq.to(config['device'])
            bseq_len = bseq_len.to(config['device'])
            btar = btar.to(config['device'])

            with autocast(device_type='cuda' if 'cuda' in config['device'] else 'cpu'):
                out = model(bseq, bseq_len, 'train')
                if isinstance(out, dict):
                    logits = out['logits']
                    pos_emb = out.get('pos_emb', None)
                    neg_emb = out.get('neg_emb', None)
                    bseq_emb = out.get('bseq_emb', None)
                    item_logits = out.get('item_logits', None)
                    basket_logits = out.get('basket_logits', None)
                    adaptive_loss = out.get('adaptive_loss', None)
                    causal_loss = out.get('causal_loss', None)
                    polar_loss = out.get('polar_loss', None)
                else:
                    logits, pos_emb, neg_emb, bseq_emb, item_logits, basket_logits = out
                    adaptive_loss, causal_loss, polar_loss = None, None, None

                rec_loss = recloss(logits, btar)
                cl_loss = cllossnp(pos_emb, neg_emb) if (pos_emb is not None and neg_emb is not None) else 0.0
                lgt_loss = logitsloss(item_logits, basket_logits) if (item_logits is not None and basket_logits is not None) else 0.0

                adp_loss = adaptive_loss if isinstance(adaptive_loss, torch.Tensor) else torch.tensor(0.0, device=logits.device)
                cau_loss = causal_loss if isinstance(causal_loss, torch.Tensor) else torch.tensor(0.0, device=logits.device)
                pol_loss = polar_loss if isinstance(polar_loss, torch.Tensor) else torch.tensor(0.0, device=logits.device)

                loss = (
                    1.25 * config['lambda_rec'] * rec_loss +
                    1.0 * config['lambda_align'] * lgt_loss +
                    config['lambda_adaptive'] * adp_loss +
                    config['lambda_causal'] * cau_loss +
                    config['lambda_polar'] * pol_loss +
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_rec += float(rec_loss if isinstance(rec_loss, float) else rec_loss.item())
            total_cl += float(cl_loss if isinstance(cl_loss, float) else (cl_loss.item() if hasattr(cl_loss, "item") else 0.0))
            total_lgt += float(lgt_loss if isinstance(lgt_loss, float) else (lgt_loss.item() if hasattr(lgt_loss, "item") else 0.0))
            total_adp += float(adp_loss if isinstance(adp_loss, float) else (adp_loss.item() if hasattr(adp_loss, "item") else 0.0))
            total_cau += float(cau_loss if isinstance(cau_loss, float) else (cau_loss.item() if hasattr(cau_loss, "item") else 0.0))
            total_pol += float(pol_loss if isinstance(pol_loss, float) else (pol_loss.item() if hasattr(pol_loss, "item") else 0.0))
            batch_count += 1

            if batch_idx % 5 == 0 or batch_idx == len(train_loader):
                print(f"[Batch {batch_idx:03d}/{len(train_loader)}] "
                      f"rec={total_rec/batch_count:.4f} | cl={total_cl/batch_count:.4f} "
                      f"| align={total_lgt/batch_count:.4f} | adp={total_adp/batch_count:.4f} | cau={total_cau/batch_count:.4f} | pol={total_pol/batch_count:.4f}")

        avg_rec = total_rec / batch_count if batch_count else 0.0
        avg_cl = total_cl / batch_count if batch_count else 0.0
        avg_lgt = total_lgt / batch_count if batch_count else 0.0
        avg_adp = total_adp / batch_count if batch_count else 0.0
        avg_cau = total_cau / batch_count if batch_count else 0.0
        avg_pol = total_pol / batch_count if batch_count else 0.0
        epoch_time = time.time() - start_time
        print(f"Epoch {e+1:03d} Summary → rec={avg_rec:.4f}, cl={avg_cl:.4f}, "
              f"align={avg_lgt:.4f}, adp={avg_adp:.4f}, cau={avg_cau:.4f}, pol={avg_pol:.4f}, "
              f"time={epoch_time:.1f}s")
        scheduler.step()

        if e % config['val_step'] == 0:
            model.eval()
            if hasattr(model, "get_graph"):
                model.get_graph(run_type='val')

            val_dataset = RecDataset(config['val_path'], lgcn_dataset.basket2id_dict, lgcn_dataset.item2id_dict)
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                collate_fn=collate_fn,
                drop_last=False,
                shuffle=False,
                **DL_KWARGS
            )

            f1s = {k: 0.0 for k in config['topk']}
            hits = {k: 0.0 for k in config['topk']}
            gts = 0
            sample_num = 0

            for _, (bseq, bseq_len, btar) in enumerate(val_loader):
                sample_num += len(bseq_len)
                gts += len(torch.nonzero(btar))
                bseq = bseq.to(config['device'])
                bseq_len = bseq_len.to(config['device'])
                btar = btar.to(config['device'])

                out = model(bseq, bseq_len, 'val')
                logits = out['logits'] if isinstance(out, dict) else out

                for k in config['topk']:
                    top_idx = torch.topk(logits, dim=1, k=k).indices
                    preds = torch.zeros_like(logits).scatter(1, top_idx, 1.0)
                    f1s[k] += sum(utils.f1(preds, btar))
                    hits[k] += utils.hit(preds, btar)

            for k in config['topk']:
                f1_val = f1s[k] / sample_num
                hr_val = hits[k] / gts
                print(f"K={k:<2} | F1={f1_val:.4f} | HR={hr_val:.4f}")

            with open(val_log_path, 'a') as f:
                for k in config['topk']:
                    res = f"Epoch {e}  K={k}  F1:{f1s[k]/sample_num:.6f}  HR:{hits[k]/gts:.6f}\n"
                    f.write(res)
                f.write("\n")

            current_combined = (
                ((f1s.get(30, 0.0) / sample_num) + (hits.get(30, 0.0) / sample_num)) / 2.0
                if sample_num else 0.0
            )

            if current_combined > best_val:
                best_val = current_combined
                count = 0
                model_save_path = f"./saved_models/{dataset}_cascade_best.pth"
                torch.save(model.state_dict(), model_save_path)
                print(f"\nBest model saved at epoch {e+1} (Avg HR@30,F1@30 = {best_val:.4f})")
            else:
                count += 1
                print(f"No improvement ({count}/{config['patience']})")
                if count >= config['patience']:
                    print("\nEarly stopping triggered.")
                    break

    return lgcn_dataset

if __name__ == "__main__":
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    dataset_tag = os.path.basename(os.path.dirname(f"./data/{args.dataset}/")) or args.dataset
    model_path = f'./saved_models/{dataset_tag}_cascade_best.pth'

    if config['run_type'] == 'train':
        val_log = f'./results/{dataset_tag}_cascade_val.txt'
        train_one_dataset(config, model_path, val_log)

    elif config['run_type'] == 'test':
        lgcn_dataset = LGCNDataset(config['train_path'], config['val_path'], config['test_path'])
        model = build_model(config, lgcn_dataset)

        if args.epoch != -1:
            load_path = f"./saved_models/{dataset_tag}_cascade_epoch{args.epoch}.pth"
            if not os.path.exists(load_path):
                print(f"Epoch {args.epoch} model not found. Using best model.")
                load_path = model_path
        else:
            load_path = model_path

        print(f"Loading model: {load_path}")
        state_dict = torch.load(load_path, map_location=config['device'], weights_only=True)
        model.load_state_dict(state_dict)

        if hasattr(model, "use_adaptive") and model.use_adaptive:
            with torch.no_grad():
                edge_probs = model.adaptive_head(
                    model.embedding_basket.weight,
                    model.embedding_item.weight
                )
                lgcn_dataset.set_adaptive_edge_weights(edge_probs)

        if hasattr(model, "use_causal") and model.use_causal:
            with torch.no_grad():
                causal_matrix = model.causal_layer.estimate_causal_matrix(
                    model.embedding_item.weight,
                    model.embedding_basket.weight
                )
                lgcn_dataset.set_causal_matrix(causal_matrix)

        if hasattr(model, "get_graph"):
            model.get_graph(run_type='test')

        out = evaluate_model(model, config['test_path'], lgcn_dataset, config)
        print(f"F1@10: {out['F1@10']:.4f}   F1@30: {out['F1@30']:.4f}   F1@50: {out['F1@50']:.4f}   "
              f"HR@10: {out['HR@10']:.4f}   HR@30: {out['HR@30']:.4f}   HR@50: {out['HR@50']:.4f}")

        with open(f'./results/{dataset_tag}_cascade_results.txt', 'w') as f:
            f.write(f"F1@10: {out['F1@10']:.6f}\nF1@30: {out['F1@30']:.6f}\nF1@50: {out['F1@50']:.6f}\n"
                    f"HR@10: {out['HR@10']:.6f}\nHR@30: {out['HR@30']:.6f}\nHR@50: {out['HR@50']:.6f}\n")

    elif config['run_type'] == 'case':
        lgcn_dataset = LGCNDataset(config['train_path'], config['val_path'], config['test_path'])
        model = build_model(config, lgcn_dataset)
        model.eval()

        if hasattr(model, "use_adaptive") and model.use_adaptive:
            with torch.no_grad():
                edge_probs = model.adaptive_head(
                    model.embedding_basket.weight,
                    model.embedding_item.weight
                )
                lgcn_dataset.set_adaptive_edge_weights(edge_probs)

        if hasattr(model, "use_causal") and model.use_causal:
            with torch.no_grad():
                causal_matrix = model.causal_layer.estimate_causal_matrix(
                    model.embedding_item.weight,
                    model.embedding_basket.weight
                )
                lgcn_dataset.set_causal_matrix(causal_matrix)

        if hasattr(model, "get_graph"):
            model.get_graph(run_type='test')

        case_dataset = RecDataset(config['case_path'], lgcn_dataset.basket2id_dict, lgcn_dataset.item2id_dict)
        case_loader = DataLoader(case_dataset, batch_size=config['batch_size'], collate_fn=collate_fn, drop_last=True, **DL_KWARGS)
        load_name = f'./saved_models/{dataset_tag}_cascade_best.pth'

        if os.path.exists(load_name):
            model.load_state_dict(torch.load(load_name, map_location=config['device']))

        for _, (bseq, bseq_len, btar) in enumerate(case_loader):
            bseq = bseq.to(config['device'])
            bseq_len = bseq_len.to(config['device'])
            _ = model(bseq, bseq_len, 'test')

        print("Case study complete.")