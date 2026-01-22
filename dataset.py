import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp
from scipy.sparse import csr_matrix


class LGCNDataset(Dataset):
    def __init__(self, train_path, val_path, test_path):
        self.trainItem = np.array([], dtype=np.int32)
        self.trainBasket = np.array([], dtype=np.int32)
        self.train_b2i_weight = np.array([], dtype=np.int32)

        self.valItem = np.array([], dtype=np.int32)
        self.valBasket = np.array([], dtype=np.int32)
        self.val_b2i_weight = np.array([], dtype=np.int32)

        self.testItem = np.array([], dtype=np.int32)
        self.testBasket = np.array([], dtype=np.int32)
        self.test_b2i_weight = np.array([], dtype=np.int32)

        self.basket2id_dict = {}
        self.item2id_dict = {}
        self.id2basket_dict = {}

        self.num_baskets = 1
        self.num_items = 1

        self.graph_cache = {}

        self.adaptive_edge_weights = None
        self.causal_matrix = None

        self._load_data(train_path, 'train')
        self._load_data(val_path, 'val')
        self._load_data(test_path, 'test')

    def _load_data(self, path, split):
        basket_arr = getattr(self, f'{split}Basket')
        item_arr = getattr(self, f'{split}Item')
        weight_arr = getattr(self, f'{split}_b2i_weight')

        with open(path, 'r') as f:
            for line in f:
                baskets = line.split('|')[:-1]

                target = baskets[-1].strip().split(' ')
                for it in target:
                    if it not in self.item2id_dict:
                        self.item2id_dict[it] = self.num_items
                        self.num_items += 1

                for b in baskets[:-1]:
                    items = sorted(self.item2id_dict.setdefault(it, self._new_item())
                                   if it in self.item2id_dict else self._add_item(it)
                                   for it in b.strip().split(' '))
                    bkey = tuple(items)

                    if bkey not in self.basket2id_dict:
                        bid = self.num_baskets
                        self.basket2id_dict[bkey] = bid
                        self.id2basket_dict[bid] = items
                        self.num_baskets += 1

                        basket_arr = np.append(basket_arr, [bid] * len(items))
                        item_arr = np.append(item_arr, items)
                        weight_arr = np.append(weight_arr, [1] * len(items))
                    else:
                        bid = self.basket2id_dict[bkey]
                        mask = basket_arr == bid
                        if mask.any():
                            weight_arr[mask] += 1
                        else:
                            basket_arr = np.append(basket_arr, [bid] * len(items))
                            item_arr = np.append(item_arr, items)
                            weight_arr = np.append(weight_arr, [1] * len(items))

        setattr(self, f'{split}Basket', basket_arr)
        setattr(self, f'{split}Item', item_arr)
        setattr(self, f'{split}_b2i_weight', weight_arr)

    def _add_item(self, it):
        self.item2id_dict[it] = self.num_items
        self.num_items += 1
        return self.item2id_dict[it]

    def _new_item(self):
        return self.num_items

    def _build_base_adj(self, split):
        if split == 'train':
            b, i, w = self.trainBasket, self.trainItem, self.train_b2i_weight
        elif split == 'val':
            b, i, w = self.valBasket, self.valItem, self.val_b2i_weight
        else:
            b, i, w = self.testBasket, self.testItem, self.test_b2i_weight

        return csr_matrix((w, (b, i)), shape=(self.num_baskets, self.num_items))

    def set_adaptive_edge_weights(self, weights):
        self.adaptive_edge_weights = weights.detach().cpu().numpy()
        self.graph_cache.clear()

    def set_causal_matrix(self, matrix):
        self.causal_matrix = matrix.detach().cpu().numpy()
        self.graph_cache.clear()

    def getSparseGraph(self, graph_type='original', run_type='train'):
        key = f'{graph_type}_{run_type}'
        if key in self.graph_cache:
            return self.graph_cache[key]

        if graph_type == 'original':
            R = self._build_base_adj(run_type)

            if run_type == 'train':
                if self.adaptive_edge_weights is not None:
                    R = R.multiply(self.adaptive_edge_weights)
                if self.causal_matrix is not None:
                    C = self.causal_matrix / (self.causal_matrix.max() + 1e-8)
                    R = R.multiply(1.0 - 0.5) + C * 0.5
        else:
            R = self._build_base_adj('train')

        adj = sp.dok_matrix((self.num_baskets + self.num_items,
                             self.num_baskets + self.num_items), dtype=np.float32)
        adj = adj.tolil()
        adj[:self.num_baskets, self.num_baskets:] = R
        adj[self.num_baskets:, :self.num_baskets] = R.T
        adj = adj.tocsr()

        rowsum = np.array(adj.sum(axis=1)).flatten()
        rowsum[rowsum == 0] = 1
        d_inv = np.power(rowsum, -0.5)
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat @ adj @ d_mat
        graph = self._to_sparse_tensor(norm_adj)
        self.graph_cache[key] = graph
        return graph

    def _to_sparse_tensor(self, mat):
        mat = mat.tocoo().astype(np.float32)
        idx = torch.tensor([mat.row, mat.col], dtype=torch.long)
        val = torch.tensor(mat.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, mat.shape).coalesce()

    def clear_cache(self):
        self.graph_cache.clear()


class RecDataset(Dataset):
    def __init__(self, path, basket2id, item2id):
        self.seqs = []
        self.lens = []
        self.tars = []

        with open(path, 'r') as f:
            for line in f:
                baskets = line.split('|')[:-1]
                seq = []
                for b in baskets[:-1]:
                    items = sorted(item2id[it] for it in b.strip().split(' '))
                    seq.append(basket2id[tuple(items)])

                self.seqs.append(seq)
                self.lens.append(len(seq))

                target = [0] * (len(item2id) + 1)
                for it in baskets[-1].strip().split(' '):
                    target[item2id[it]] = 1
                self.tars.append(target)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.seqs[idx]),
            torch.tensor(self.lens[idx]),
            torch.tensor(self.tars[idx])
        )
