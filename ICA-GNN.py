#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import math
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== CONFIG ==================

INPUT_XLSX = "/lustre/home/acct-clsbfw/zx2018/HXY/MetaBiogas/summary/Analysis/251211_cluster/GNN-ICA/GNN-Database.xlsx"
RA_SHEET_NAME = "RA"
INTERACTION_SHEET_NAME = "Interaction"

OUT_DIR = "/lustre/home/acct-clsbfw/zx2018/HXY/MetaBiogas/summary/Analysis/251211_cluster/GNN-ICA"
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(OUT_DIR, "GNN_ICA_RedSup_Summary.csv")

N_BOOTSTRAP = 50
ICA_N_COMPONENTS_LIST = [1,2, 3, 4, 5, 6]
GNN_HIDDEN_DIM_LIST = [32, 64]
GNN_NUM_LAYERS_LIST = [1, 2]

K_LIST = [5, 6, 7, 8, 9, 10]
KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 1000

GNN_LR = 1e-3
GNN_EPOCHS = 500

GLOBAL_RANDOM_SEED = 20251211

MIN_N_CLUSTERS = 2
MIN_N_NON_SINGLETON_CLUSTERS = 5


# ================== 小工具 ==================


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def minmax_scale_for_edges(mat: np.ndarray) -> np.ndarray:
    mat = mat.copy().astype(np.float32)
    N = mat.shape[0]
    mask_offdiag = ~np.eye(N, dtype=bool)
    vals = mat[mask_offdiag & (mat > 0)]
    if vals.size == 0:
        return np.zeros_like(mat, dtype=np.float32)
    v_min = vals.min()
    v_max = vals.max()
    if v_max <= v_min:
        scaled = np.zeros_like(mat, dtype=np.float32)
        scaled[mat > 0] = 1.0
        return scaled
    scaled = (mat - v_min) / (v_max - v_min)
    scaled[mat == 0] = 0.0
    return scaled.astype(np.float32)


def build_comp_compl_matrices(df_interaction: pd.DataFrame, genomes: list):
    genome_to_idx = {g: i for i, g in enumerate(genomes)}
    N = len(genomes)
    comp_mat = np.zeros((N, N), dtype=np.float32)
    compl_mat = np.zeros((N, N), dtype=np.float32)

    for _, row in df_interaction.iterrows():
        g1 = row["Genome1"]
        g2 = row["Genome2"]
        if g1 not in genome_to_idx or g2 not in genome_to_idx:
            continue
        i = genome_to_idx[g1]
        j = genome_to_idx[g2]
        if i == j:
            continue
        comp = row.get("Competition_Ave", np.nan)
        compl = row.get("Complementarity_Ave", np.nan)
        if pd.notnull(comp):
            comp = float(comp)
            comp_mat[i, j] = comp
            comp_mat[j, i] = comp
        if pd.notnull(compl):
            compl = float(compl)
            compl_mat[i, j] = compl
            compl_mat[j, i] = compl

    np.fill_diagonal(comp_mat, 0.0)
    np.fill_diagonal(compl_mat, 0.0)
    return comp_mat, compl_mat


def compute_baseline_stats(comp_mat: np.ndarray, compl_mat: np.ndarray):
    """
    compute comp_all, compl_all
    """
    N = comp_mat.shape[0]
    mask_offdiag = ~np.eye(N, dtype=bool)
    comp_vals = comp_mat[mask_offdiag]
    compl_vals = compl_mat[mask_offdiag]

    comp_all = float(comp_vals.mean()) if comp_vals.size > 0 else 0.0
    compl_all = float(compl_vals.mean()) if compl_vals.size > 0 else 0.0

    if comp_all <= 0:
        comp_all = 1e-8
    if compl_all <= 0:
        compl_all = 1e-8

    return comp_all, compl_all


def build_normalized_adj(edge_mat: np.ndarray) -> np.ndarray:
    """
        A_tilde = A + I
        D = diag(sum_j A_tilde_ij)
        A_norm = D^{-1/2} A_tilde D^{-1/2}
    """
    N = edge_mat.shape[0]
    A = edge_mat.copy().astype(np.float32)
    A = A + np.eye(N, dtype=np.float32)
    deg = A.sum(axis=1)
    deg[deg <= 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm.astype(np.float32)


def run_ica_embedding(X_ra_boot: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_ra_boot)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ica = FastICA(
            n_components=n_components,
            max_iter=1000,
            random_state=random_state,
            whiten="unit-variance",
        )
        Z = ica.fit_transform(X_scaled)

    return Z.astype(np.float32)


# ================== GNN ==================


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim)
        h = self.linear(x)          # (N, out_dim)
        h = torch.matmul(A_norm, h) # (N, out_dim)
        return h


class GNNFeatureAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GCNLayer(in_dim, hidden_dim))
        else:
            self.layers.append(GCNLayer(in_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(GCNLayer(hidden_dim, hidden_dim))

        self.decoder = nn.Linear(hidden_dim, in_dim)

    def encode(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, A_norm)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor):
        z = self.encode(x, A_norm)
        x_hat = self.decoder(z)
        return x_hat, z


def train_gnn_autoencoder(
    X_node: np.ndarray,
    A_norm: np.ndarray,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    lr: float,
    device: torch.device,
    random_state: int,
) -> np.ndarray:
    set_global_seed(random_state)

    x = torch.from_numpy(X_node).float().to(device)      # (N, in_dim)
    A_t = torch.from_numpy(A_norm).float().to(device)    # (N, N)

    model = GNNFeatureAutoEncoder(X_node.shape[1], hidden_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_hat, z = model(x, A_t)
        loss = F.mse_loss(x_hat, x)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, z_final = model(x, A_t)

    z_np = z_final.cpu().numpy().astype(np.float32)
    return z_np


# ================== clustering and evaluation ==================


def compute_cluster_edge_stats(labels: np.ndarray,
                               comp_mat: np.ndarray,
                               compl_mat: np.ndarray):
    comp_sum = 0.0
    compl_sum = 0.0
    total_pairs = 0

    labels = np.asarray(labels)
    unique_clusters = np.unique(labels)

    for c in unique_clusters:
        idx = np.where(labels == c)[0]
        size = idx.size
        if size < 2:
            continue
        sub_comp = comp_mat[np.ix_(idx, idx)]
        sub_compl = compl_mat[np.ix_(idx, idx)]
        mask_offdiag = ~np.eye(size, dtype=bool)
        comp_vals = sub_comp[mask_offdiag]
        compl_vals = sub_compl[mask_offdiag]
        comp_sum += float(comp_vals.sum())
        compl_sum += float(compl_vals.sum())
        total_pairs += comp_vals.size

    if total_pairs == 0:
        return math.nan, math.nan

    comp_in = comp_sum / total_pairs
    compl_in = compl_sum / total_pairs
    return comp_in, compl_in


def evaluate_clustering(Z: np.ndarray,
                        labels: np.ndarray,
                        model_type: str,
                        comp_mat: np.ndarray,
                        compl_mat: np.ndarray,
                        comp_all: float,
                        compl_all: float):
    if len(np.unique(labels)) < 2:
        return None

    try:
        sli = silhouette_score(Z, labels, metric="euclidean")
    except Exception:
        return None

    uniq, counts = np.unique(labels, return_counts=True)
    n_clusters = uniq.size
    n_non_singleton = int((counts >= 2).sum())
    if n_clusters < MIN_N_CLUSTERS or n_non_singleton < MIN_N_NON_SINGLETON_CLUSTERS:
        return None

    comp_in, compl_in = compute_cluster_edge_stats(labels, comp_mat, compl_mat)
    if math.isnan(comp_in) or math.isnan(compl_in):
        return None

    if model_type == "Red":
        award = comp_in / comp_all
        ratio_compl = compl_in / compl_all
        if ratio_compl <= 1.0:
            penalty = 1.0
        else:
            penalty = compl_all / compl_in
    elif model_type == "Sup":
        award = compl_in / compl_all
        ratio_comp = comp_in / comp_all
        if ratio_comp <= 1.0:
            penalty = 1.0
        else:
            penalty = comp_all / comp_in
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    sli_award = sli * award
    sli_award_penalty = sli * award * penalty

    return {
        "sli": float(sli),
        "comp_in": float(comp_in),
        "compl_in": float(compl_in),
        "award": float(award),
        "penalty": float(penalty),
        "sli_award": float(sli_award),
        "sli_award_penalty": float(sli_award_penalty),
    }


# ================== main ==================


def main():
    device = torch.device("cpu")
    df_ra = pd.read_excel(INPUT_XLSX, sheet_name=RA_SHEET_NAME, index_col=0)
    df_inter = pd.read_excel(INPUT_XLSX, sheet_name=INTERACTION_SHEET_NAME)

    genomes = df_ra.index.tolist()
    X_ra_raw = df_ra.values.astype(np.float32)
    N_genomes, N_samples = X_ra_raw.shape
    X_ra_log = np.log1p(X_ra_raw)
    comp_mat_raw, compl_mat_raw = build_comp_compl_matrices(df_inter, genomes)
    comp_all, compl_all = compute_baseline_stats(comp_mat_raw, compl_mat_raw)


    comp_edge_single = minmax_scale_for_edges(comp_mat_raw)
    compl_edge_single = minmax_scale_for_edges(compl_mat_raw)

    red_diff_edge = np.maximum(comp_edge_single - compl_edge_single, 0.0).astype(np.float32)
    np.fill_diagonal(red_diff_edge, 0.0)

    sup_diff_edge = np.maximum(compl_edge_single - comp_edge_single, 0.0).astype(np.float32)
    np.fill_diagonal(sup_diff_edge, 0.0)

    A_norm_dict = {}
    A_norm_dict[("Red", "single")] = build_normalized_adj(comp_edge_single)
    A_norm_dict[("Red", "diff")] = build_normalized_adj(red_diff_edge)
    A_norm_dict[("Sup", "single")] = build_normalized_adj(compl_edge_single)
    A_norm_dict[("Sup", "diff")] = build_normalized_adj(sup_diff_edge)

    model_types = ["Red", "Sup"]
    edge_types = ["single", "diff"]

    # key: (model_type, edge_type, ica_n_components, gnn_hidden_dim, gnn_num_layers)
    results = defaultdict(lambda: {
        "sli_list": [],
        "sli_award_list": [],
        "sli_award_penalty_list": [],
        "comp_in_list": [],
        "compl_in_list": [],
        "best_K_list": [],
    })

    rng = np.random.default_rng(GLOBAL_RANDOM_SEED)

    total_ica_tasks = len(ICA_N_COMPONENTS_LIST) * N_BOOTSTRAP
    task_counter = 0

    for ica_n_components in ICA_N_COMPONENTS_LIST:
        print(f"\n[INFO] ===== ICA n_components = {ica_n_components} =====")
        for b_idx in range(N_BOOTSTRAP):
            task_counter += 1
            print(f"[INFO] Bootstrap {b_idx + 1}/{N_BOOTSTRAP} (global {task_counter}/{total_ica_tasks}) ...")

            # --- 3.1 RA bootstrap ---
            sample_indices = rng.integers(low=0, high=N_samples, size=N_samples, endpoint=False)
            X_ra_boot = X_ra_log[:, sample_indices]

            # --- 3.2 ICA embedding ---
            ica_seed = GLOBAL_RANDOM_SEED + 1000 * ica_n_components + b_idx
            Z_ica = run_ica_embedding(X_ra_boot, n_components=ica_n_components, random_state=ica_seed)
            # Z_ica: (N_genomes, ica_n_components)

            for gnn_hidden_dim in GNN_HIDDEN_DIM_LIST:
                for gnn_num_layers in GNN_NUM_LAYERS_LIST:
                    gnn_seed_base = (GLOBAL_RANDOM_SEED +
                                     100000 * ica_n_components +
                                     1000 * gnn_hidden_dim +
                                     100 * gnn_num_layers +
                                     b_idx)

                    for model_type in model_types:
                        for edge_type in edge_types:
                            A_norm = A_norm_dict[(model_type, edge_type)]
                            gnn_seed = gnn_seed_base + (0 if model_type == "Red" else 10) + (0 if edge_type == "single" else 1)

                            Z_gnn = train_gnn_autoencoder(
                                X_node=Z_ica,
                                A_norm=A_norm,
                                hidden_dim=gnn_hidden_dim,
                                num_layers=gnn_num_layers,
                                epochs=GNN_EPOCHS,
                                lr=GNN_LR,
                                device=device,
                                random_state=gnn_seed,
                            )
                            Z_cluster = Z_gnn
                            best_metrics = None
                            best_K = None

                            for K in K_LIST:
                                try:
                                    km = KMeans(
                                        n_clusters=K,
                                        n_init=KMEANS_N_INIT,
                                        max_iter=KMEANS_MAX_ITER,
                                        random_state=gnn_seed + K,
                                    )
                                    labels = km.fit_predict(Z_cluster)
                                except Exception:
                                    continue

                                metrics = evaluate_clustering(
                                    Z=Z_cluster,
                                    labels=labels,
                                    model_type=model_type,
                                    comp_mat=comp_mat_raw,
                                    compl_mat=compl_mat_raw,
                                    comp_all=comp_all,
                                    compl_all=compl_all,
                                )
                                if metrics is None:
                                    continue

                                if best_metrics is None or metrics["sli"] > best_metrics["sli"]:
                                    best_metrics = metrics
                                    best_K = K

                            if best_metrics is not None and best_K is not None:
                                key = (model_type, edge_type, ica_n_components, gnn_hidden_dim, gnn_num_layers)
                                res = results[key]
                                res["sli_list"].append(best_metrics["sli"])
                                res["sli_award_list"].append(best_metrics["sli_award"])
                                res["sli_award_penalty_list"].append(best_metrics["sli_award_penalty"])
                                res["comp_in_list"].append(best_metrics["comp_in"])
                                res["compl_in_list"].append(best_metrics["compl_in"])
                                res["best_K_list"].append(best_K)


    rows = []
    for key, val in results.items():
        model_type, edge_type, ica_n_components, gnn_hidden_dim, gnn_num_layers = key
        sli_arr = np.array(val["sli_list"], dtype=float)
        sli_award_arr = np.array(val["sli_award_list"], dtype=float)
        sli_award_penalty_arr = np.array(val["sli_award_penalty_list"], dtype=float)
        comp_in_arr = np.array(val["comp_in_list"], dtype=float)
        compl_in_arr = np.array(val["compl_in_list"], dtype=float)
        best_K_list = val["best_K_list"]

        n_valid = sli_arr.size
        if n_valid == 0:
            sli_mean = float("nan")
            sli_std = float("nan")
            sli_award_mean = float("nan")
            sli_award_penalty_mean = float("nan")
            comp_in_mean = float("nan")
            compl_in_mean = float("nan")
            best_K_mode = None
            best_K_mean = float("nan")
        else:
            sli_mean = float(sli_arr.mean())
            sli_std = float(sli_arr.std(ddof=1)) if n_valid > 1 else 0.0
            sli_award_mean = float(sli_award_arr.mean())
            sli_award_penalty_mean = float(sli_award_penalty_arr.mean())
            comp_in_mean = float(comp_in_arr.mean())
            compl_in_mean = float(compl_in_arr.mean())
            best_K_mean = float(np.mean(best_K_list))
            try:
                best_K_mode = Counter(best_K_list).most_common(1)[0][0]
            except Exception:
                best_K_mode = None

        rows.append({
            "Model_Type": model_type,
            "Edge_Type": edge_type,
            "ICA_n_components": ica_n_components,
            "GNN_hidden_dim": gnn_hidden_dim,
            "GNN_num_layers": gnn_num_layers,
            "N_bootstrap_total": N_BOOTSTRAP,
            "N_bootstrap_valid": n_valid,
            "Sli_mean": sli_mean,
            "Sli_std": sli_std,
            "Sli_Award_mean": sli_award_mean,
            "Sli_AwardPenalty_mean": sli_award_penalty_mean,
            "Comp_in_mean": comp_in_mean,
            "Compl_in_mean": compl_in_mean,
            "Best_K_mode": best_K_mode,
            "Best_K_mean": best_K_mean,
        })

    df_summary = pd.DataFrame(rows)
    df_summary = df_summary.sort_values(
        by=["Model_Type", "Edge_Type", "ICA_n_components", "GNN_hidden_dim", "GNN_num_layers"]
    )
    df_summary.to_csv(SUMMARY_CSV, index=False)


if __name__ == "__main__":
    main()

