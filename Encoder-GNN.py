#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================== CONFIG ==================

INPUT_XLSX = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/251211-Cluster/GNN-Encoder/GNN-Database.xlsx"
RA_SHEET_NAME = "RA"
INTERACTION_SHEET_NAME = "Interaction"

OUT_DIR = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/251211-Cluster/GNN-Encoder/GNN-Encoder"
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(OUT_DIR, "GNN_Encoder_RedSup_Summary.csv")

# Hyperparameters setting
BOOTSTRAP_INITIAL = 1
BOOTSTRAP_MAX = 20
SLI_TRIGGER_THRESHOLD = 0.40

ENC_LATENT_DIM_LIST = [2, 3, 4]
ENC_HIDDEN_DIM_LIST = [64, 128, 256]

ENC_LR = 1e-3
ENC_EPOCHS = 500
ENC_WEIGHT_DECAY = 0.0
ENC_DROPOUT = 0.0


GNN_HIDDEN_DIM_LIST = [32, 64, 128]
GNN_NUM_LAYERS_LIST = [1, 2, 3]

K_LIST = [5, 6, 7, 8, 9, 10]
KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 1000

GNN_LR = 1e-3
GNN_EPOCHS = 500

GLOBAL_RANDOM_SEED = 20251211

MIN_N_CLUSTERS = 2
MIN_N_NON_SINGLETON_CLUSTERS = 5



def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def bootstrap_sample_indices(base_seed: int, b_idx: int, n_samples: int) -> np.ndarray:
    rng = np.random.default_rng(base_seed + b_idx)
    return rng.integers(low=0, high=n_samples, size=n_samples, endpoint=False)


def minmax_scale_for_edges(mat: np.ndarray) -> np.ndarray:
    mat = mat.copy().astype(np.float32)
    N = mat.shape[0]
    mask_offdiag = ~np.eye(N, dtype=bool)
    vals = mat[mask_offdiag]
    pos = vals[vals > 0]
    if pos.size == 0:
        return mat
    vmin, vmax = float(pos.min()), float(pos.max())
    if vmax <= vmin:
        return mat
    mat[mask_offdiag] = np.where(mat[mask_offdiag] > 0, (mat[mask_offdiag] - vmin) / (vmax - vmin), 0.0)
    mat = np.clip(mat, 0.0, 1.0)
    return mat.astype(np.float32)


def build_normalized_adj(edge_mat: np.ndarray) -> np.ndarray:
    N = edge_mat.shape[0]
    A = edge_mat.copy().astype(np.float32)
    A = A + np.eye(N, dtype=np.float32)
    deg = A.sum(axis=1)
    deg[deg <= 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm.astype(np.float32)


# ================== Encoder (MLP AutoEncoder) ==================

class MLPAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def run_encoder_embedding(
    X_ra_boot: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    random_state: int,
    device: torch.device,
) -> np.ndarray:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_ra_boot).astype(np.float32)

    set_global_seed(random_state)

    x = torch.from_numpy(X_scaled).to(device)
    model = MLPAutoEncoder(input_dim=x.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=ENC_DROPOUT).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=ENC_LR, weight_decay=ENC_WEIGHT_DECAY)

    model.train()
    for _ in range(ENC_EPOCHS):
        opt.zero_grad(set_to_none=True)
        x_hat, _ = model(x)
        loss = F.mse_loss(x_hat, x)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        _, z = model(x)

    return z.detach().cpu().numpy().astype(np.float32)


def build_comp_compl_matrices(df_inter: pd.DataFrame, genomes: list):
    needed = {"Genome1", "Genome2", "Competition_Ave", "Complementarity_Ave"}
    missing = needed - set(df_inter.columns)
    if missing:
        raise ValueError(f"Interaction sheet miss col: {sorted(list(missing))}")

    idx_map = {g: i for i, g in enumerate(genomes)}
    N = len(genomes)
    comp_mat = np.zeros((N, N), dtype=np.float32)
    compl_mat = np.zeros((N, N), dtype=np.float32)

    for _, row in df_inter.iterrows():
        g1 = str(row["Genome1"])
        g2 = str(row["Genome2"])
        if g1 not in idx_map or g2 not in idx_map:
            continue
        i, j = idx_map[g1], idx_map[g2]
        comp = row["Competition_Ave"]
        compl = row["Complementarity_Ave"]

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
    N = comp_mat.shape[0]
    mask_offdiag = ~np.eye(N, dtype=bool)
    comp_vals = comp_mat[mask_offdiag]
    compl_vals = compl_mat[mask_offdiag]
    comp_all = float(comp_vals.mean()) if comp_vals.size else 0.0
    compl_all = float(compl_vals.mean()) if compl_vals.size else 0.0
    return comp_all, compl_all

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = torch.matmul(A_norm, h)
        return h


class GNNFeatureAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

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
    features: np.ndarray,
    A_norm: np.ndarray,
    hidden_dim: int,
    num_layers: int,
    lr: float,
    epochs: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    set_global_seed(seed)

    x = torch.from_numpy(features.astype(np.float32)).to(device)
    A_t = torch.from_numpy(A_norm.astype(np.float32)).to(device)

    model = GNNFeatureAutoEncoder(in_dim=x.shape[1], hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        x_hat, _ = model(x, A_t)
        loss = F.mse_loss(x_hat, x)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, z_final = model(x, A_t)

    return z_final.detach().cpu().numpy().astype(np.float32)


# ================== Clustering and evaluation ==================

def compute_cluster_edge_stats(labels: np.ndarray, comp_mat: np.ndarray, compl_mat: np.ndarray):

    N = labels.size
    clusters = np.unique(labels)
    total_pairs = 0
    comp_sum = 0.0
    compl_sum = 0.0

    for c in clusters:
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

    return comp_sum / total_pairs, compl_sum / total_pairs


def count_non_singleton_clusters(labels: np.ndarray) -> int:
    cnt = 0
    for c in np.unique(labels):
        if np.sum(labels == c) >= 2:
            cnt += 1
    return cnt


def evaluate_clustering(
    Z: np.ndarray,
    labels: np.ndarray,
    model_type: str,
    comp_mat: np.ndarray,
    compl_mat: np.ndarray,
    comp_all: float,
    compl_all: float,
):

    n_clusters = len(np.unique(labels))
    if n_clusters < MIN_N_CLUSTERS:
        return None
    n_non_singleton = count_non_singleton_clusters(labels)
    if n_non_singleton < MIN_N_NON_SINGLETON_CLUSTERS:
        return None

    # silhouette
    try:
        sli = silhouette_score(Z, labels, metric="euclidean")
    except Exception:
        return None

    comp_in, compl_in = compute_cluster_edge_stats(labels, comp_mat, compl_mat)
    if math.isnan(comp_in) or math.isnan(compl_in):
        return None

    eps = 1e-12
    if model_type == "Red":
        award = (comp_in + eps) / (comp_all + eps)
        penalty = 1.0 / ((compl_in + eps) / (compl_all + eps))
    else:
        award = (compl_in + eps) / (compl_all + eps)
        penalty = 1.0 / ((comp_in + eps) / (comp_all + eps))

    award = float(np.clip(award, 0.0, 10.0))
    penalty = float(np.clip(penalty, 0.0, 10.0))

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



def main():
    device = get_device()
    print(f"[INFO] Using device: {device}")

    df_ra = pd.read_excel(INPUT_XLSX, sheet_name=RA_SHEET_NAME, index_col=0)
    df_inter = pd.read_excel(INPUT_XLSX, sheet_name=INTERACTION_SHEET_NAME)

    genomes = df_ra.index.tolist()
    X_ra_raw = df_ra.values.astype(np.float32)
    N_genomes, N_samples = X_ra_raw.shape

    # log1p
    X_ra_log = np.log1p(X_ra_raw)

    comp_mat_raw, compl_mat_raw = build_comp_compl_matrices(df_inter, genomes)
    comp_all, compl_all = compute_baseline_stats(comp_mat_raw, compl_mat_raw)
    comp_edge_single = minmax_scale_for_edges(comp_mat_raw)
    compl_edge_single = minmax_scale_for_edges(compl_mat_raw)

    red_diff_edge = comp_edge_single - compl_edge_single
    red_diff_edge[red_diff_edge < 0] = 0.0
    np.fill_diagonal(red_diff_edge, 0.0)

    sup_diff_edge = compl_edge_single - comp_edge_single
    sup_diff_edge[sup_diff_edge < 0] = 0.0
    np.fill_diagonal(sup_diff_edge, 0.0)

    A_norm_dict = {
        ("Red", "single"): build_normalized_adj(comp_edge_single),
        ("Red", "diff"): build_normalized_adj(red_diff_edge),
        ("Sup", "single"): build_normalized_adj(compl_edge_single),
        ("Sup", "diff"): build_normalized_adj(sup_diff_edge),
    }


    model_types = ["Red", "Sup"]
    edge_types = ["single", "diff"]

    # key: (model_type, edge_type, enc_latent_dim, enc_hidden_dim, gnn_hidden_dim, gnn_num_layers)
    results = defaultdict(lambda: {
        "sli_list": [],
        "sli_award_list": [],
        "sli_award_penalty_list": [],
        "comp_in_list": [],
        "compl_in_list": [],
        "best_K_list": [],
        "n_attempted": 0,
        "n_planned": BOOTSTRAP_INITIAL,
        "trigger_sli_1st": float("nan"),
        "triggered": False,
    })

    total_stage1 = len(ENC_LATENT_DIM_LIST) * len(ENC_HIDDEN_DIM_LIST) * len(GNN_HIDDEN_DIM_LIST) * len(GNN_NUM_LAYERS_LIST) * len(model_types) * len(edge_types)
    stage1_done = 0

    for enc_latent in ENC_LATENT_DIM_LIST:
        for enc_hidden in ENC_HIDDEN_DIM_LIST:
            print(f"\n[INFO] ===== Encoder latent={enc_latent}, hidden={enc_hidden} =====")

            # --- Stage 1: b_idx = 0 ---
            b_idx = 0
            sample_seed_base = GLOBAL_RANDOM_SEED + 100000 * enc_latent + 1000 * enc_hidden
            sample_indices = bootstrap_sample_indices(sample_seed_base, b_idx, N_samples)
            X_ra_boot = X_ra_log[:, sample_indices]

            enc_seed = GLOBAL_RANDOM_SEED + 100000 * enc_latent + 1000 * enc_hidden + b_idx
            Z_enc = run_encoder_embedding(
                X_ra_boot=X_ra_boot,
                latent_dim=enc_latent,
                hidden_dim=enc_hidden,
                random_state=enc_seed,
                device=device,
            )

            triggered_keys = set()

            for gnn_hidden_dim in GNN_HIDDEN_DIM_LIST:
                for gnn_num_layers in GNN_NUM_LAYERS_LIST:
                    gnn_seed_base = (GLOBAL_RANDOM_SEED +
                                     100000 * enc_latent + 1000 * enc_hidden +
                                     100 * gnn_hidden_dim + 10 * gnn_num_layers)

                    for model_type in model_types:
                        for edge_type in edge_types:
                            stage1_done += 1
                            print(f"[STAGE1] {stage1_done}/{total_stage1} enc({enc_latent},{enc_hidden}) "
                                  f"gnn(h={gnn_hidden_dim},L={gnn_num_layers}) {model_type}-{edge_type}")

                            key = (model_type, edge_type, enc_latent, enc_hidden, gnn_hidden_dim, gnn_num_layers)
                            A_norm = A_norm_dict[(model_type, edge_type)]
                            gnn_seed = gnn_seed_base + b_idx + (0 if model_type == "Red" else 5000) + (0 if edge_type == "single" else 7000)

                            Z_cluster = train_gnn_autoencoder(
                                features=Z_enc,
                                A_norm=A_norm,
                                hidden_dim=gnn_hidden_dim,
                                num_layers=gnn_num_layers,
                                lr=GNN_LR,
                                epochs=GNN_EPOCHS,
                                seed=gnn_seed,
                                device=device,
                            )

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

                            results[key]["n_attempted"] += 1

                            if best_metrics is not None:
                                results[key]["sli_list"].append(best_metrics["sli"])
                                results[key]["sli_award_list"].append(best_metrics["sli_award"])
                                results[key]["sli_award_penalty_list"].append(best_metrics["sli_award_penalty"])
                                results[key]["comp_in_list"].append(best_metrics["comp_in"])
                                results[key]["compl_in_list"].append(best_metrics["compl_in"])
                                results[key]["best_K_list"].append(best_K if best_K is not None else -1)

                                results[key]["trigger_sli_1st"] = best_metrics["sli"]

                                if best_metrics["sli"] > SLI_TRIGGER_THRESHOLD:
                                    results[key]["n_planned"] = BOOTSTRAP_MAX
                                    results[key]["triggered"] = True
                                    triggered_keys.add(key)
                            else:
                                results[key]["trigger_sli_1st"] = float("nan")

            if not triggered_keys:
                print("[INFO] invalidate combination")
                continue

            print(f"[INFO] {len(triggered_keys)} / "
                  f"{len(GNN_HIDDEN_DIM_LIST) * len(GNN_NUM_LAYERS_LIST) * len(model_types) * len(edge_types)}")
            for b_idx in range(1, BOOTSTRAP_MAX):
                sample_indices = bootstrap_sample_indices(sample_seed_base, b_idx, N_samples)
                X_ra_boot = X_ra_log[:, sample_indices]

                enc_seed = GLOBAL_RANDOM_SEED + 100000 * enc_latent + 1000 * enc_hidden + b_idx
                Z_enc = run_encoder_embedding(
                    X_ra_boot=X_ra_boot,
                    latent_dim=enc_latent,
                    hidden_dim=enc_hidden,
                    random_state=enc_seed,
                    device=device,
                )

                for gnn_hidden_dim in GNN_HIDDEN_DIM_LIST:
                    for gnn_num_layers in GNN_NUM_LAYERS_LIST:
                        gnn_seed_base = (GLOBAL_RANDOM_SEED +
                                         100000 * enc_latent + 1000 * enc_hidden +
                                         100 * gnn_hidden_dim + 10 * gnn_num_layers)

                        for model_type in model_types:
                            for edge_type in edge_types:
                                key = (model_type, edge_type, enc_latent, enc_hidden, gnn_hidden_dim, gnn_num_layers)
                                if key not in triggered_keys:
                                    continue

                                A_norm = A_norm_dict[(model_type, edge_type)]
                                gnn_seed = gnn_seed_base + b_idx + (0 if model_type == "Red" else 5000) + (0 if edge_type == "single" else 7000)

                                Z_cluster = train_gnn_autoencoder(
                                    features=Z_enc,
                                    A_norm=A_norm,
                                    hidden_dim=gnn_hidden_dim,
                                    num_layers=gnn_num_layers,
                                    lr=GNN_LR,
                                    epochs=GNN_EPOCHS,
                                    seed=gnn_seed,
                                    device=device,
                                )

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

                                results[key]["n_attempted"] += 1

                                if best_metrics is not None:
                                    results[key]["sli_list"].append(best_metrics["sli"])
                                    results[key]["sli_award_list"].append(best_metrics["sli_award"])
                                    results[key]["sli_award_penalty_list"].append(best_metrics["sli_award_penalty"])
                                    results[key]["comp_in_list"].append(best_metrics["comp_in"])
                                    results[key]["compl_in_list"].append(best_metrics["compl_in"])
                                    results[key]["best_K_list"].append(best_K if best_K is not None else -1)


    rows = []
    for key, val in results.items():
        model_type, edge_type, enc_latent, enc_hidden, gnn_hidden_dim, gnn_num_layers = key

        sli_arr = np.array(val["sli_list"], dtype=float)
        sli_award_arr = np.array(val["sli_award_list"], dtype=float)
        sli_award_penalty_arr = np.array(val["sli_award_penalty_list"], dtype=float)
        comp_in_arr = np.array(val["comp_in_list"], dtype=float)
        compl_in_arr = np.array(val["compl_in_list"], dtype=float)
        bestK_arr = np.array(val["best_K_list"], dtype=int)

        n_attempted = int(val["n_attempted"])
        n_planned = int(val["n_planned"])
        n_valid = int(sli_arr.size)

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
            try:
                vals_k, counts_k = np.unique(bestK_arr[bestK_arr >= 0], return_counts=True)
                best_K_mode = int(vals_k[np.argmax(counts_k)]) if vals_k.size else None
            except Exception:
                best_K_mode = None
            best_K_mean = float(bestK_arr[bestK_arr >= 0].mean()) if np.any(bestK_arr >= 0) else float("nan")

        rows.append({
            "Model_Type": model_type,
            "Edge_Type": edge_type,
            "ENC_latent_dim": enc_latent,
            "ENC_hidden_dim": enc_hidden,
            "GNN_hidden_dim": gnn_hidden_dim,
            "GNN_num_layers": gnn_num_layers,
            "Bootstrap_planned": n_planned,
            "Bootstrap_attempted": n_attempted,
            "Bootstrap_valid": n_valid,
            "Trigger_sli_1st": val["trigger_sli_1st"],
            "Triggered_expand": bool(val["triggered"]),
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
        by=["Model_Type", "Edge_Type", "ENC_latent_dim", "ENC_hidden_dim", "GNN_hidden_dim", "GNN_num_layers"]
    )
    df_summary.to_csv(SUMMARY_CSV, index=False)


if __name__ == "__main__":
    main()
