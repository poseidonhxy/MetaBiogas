#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import math
from typing import Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F



# CONFIG
INPUT_XLSX = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/251211-Cluster/GNN-Encoder/GNN-Database.xlsx"
RA_SHEET_NAME = "RA"
INTERACTION_SHEET_NAME = "Interaction"

OUT_DIR = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/251211-Cluster/GNN-Encoder/Final_Red_Single_SA_SAP"
os.makedirs(OUT_DIR, exist_ok=True)

GLOBAL_RANDOM_SEED = 20251211
SAVE_EMBEDDING = True

USE_BOOTSTRAP = False
BOOTSTRAP_B_IDX = 5

MODEL_TYPE = "Red"      # "Red" or "Sup"
EDGE_TYPE  = "single"   # "single" or "diff"

# ---- Encoder(AE)  ----
ENC_LATENT_DIM   = 3
ENC_HIDDEN_DIM   = 128
ENC_LR           = 1e-3
ENC_EPOCHS       = 500
ENC_WEIGHT_DECAY = 0.0
ENC_DROPOUT      = 0.0

# ---- GNN(AE)  ----
GNN_HIDDEN_DIM  = 32
GNN_NUM_LAYERS  = 2
GNN_LR          = 1e-3
GNN_EPOCHS      = 500

# ---- KMeans----
K = 5
KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 1000

MIN_N_CLUSTERS = 2
MIN_N_NON_SINGLETON_CLUSTERS = 5
# ==========================================================


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)

def count_non_singleton_clusters(labels: np.ndarray) -> int:
    cnt = 0
    for c in np.unique(labels):
        if np.sum(labels == c) >= 2:
            cnt += 1
    return cnt

def compute_cluster_edge_stats(labels: np.ndarray, comp_mat: np.ndarray, compl_mat: np.ndarray) -> Tuple[float, float]:
    total_pairs = 0
    comp_sum = 0.0
    compl_sum = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if idx.size < 2:
            continue
        sub_comp = comp_mat[np.ix_(idx, idx)]
        sub_compl = compl_mat[np.ix_(idx, idx)]
        m = ~np.eye(idx.size, dtype=bool)
        comp_sum += float(sub_comp[m].sum())
        compl_sum += float(sub_compl[m].sum())
        total_pairs += int(m.sum())
    if total_pairs == 0:
        return math.nan, math.nan
    return comp_sum / total_pairs, compl_sum / total_pairs

def evaluate_clustering(Z: np.ndarray,
                        labels: np.ndarray,
                        model_type: str,
                        comp_mat: np.ndarray,
                        compl_mat: np.ndarray,
                        comp_all: float,
                        compl_all: float) -> Dict[str, float]:
    n_clusters = len(np.unique(labels))
    if n_clusters < MIN_N_CLUSTERS:
        raise RuntimeError(f"Invalid clustering: n_clusters={n_clusters} < {MIN_N_CLUSTERS}")
    n_non_singleton = count_non_singleton_clusters(labels)
    if n_non_singleton < MIN_N_NON_SINGLETON_CLUSTERS:
        raise RuntimeError(f"Invalid clustering: n_non_singleton_clusters={n_non_singleton} < {MIN_N_NON_SINGLETON_CLUSTERS}")

    sli = float(silhouette_score(Z, labels, metric="euclidean"))

    comp_in, compl_in = compute_cluster_edge_stats(labels, comp_mat, compl_mat)
    if math.isnan(comp_in) or math.isnan(compl_in):
        raise RuntimeError("Invalid clustering: no within-cluster pairs for edge stats.")

    eps = 1e-12
    if model_type == "Red":
        award = (comp_in + eps) / (comp_all + eps)
        penalty = 1.0 / ((compl_in + eps) / (compl_all + eps))
    else:
        award = (compl_in + eps) / (compl_all + eps)
        penalty = 1.0 / ((comp_in + eps) / (comp_all + eps))

    award = float(np.clip(award, 0.0, 10.0))
    penalty = float(np.clip(penalty, 0.0, 10.0))

    return {
        "sli": sli,
        "comp_in": float(comp_in),
        "compl_in": float(compl_in),
        "award": float(award),
        "penalty": float(penalty),
        "sli_award": float(sli * award),
        "sli_award_penalty": float(sli * award * penalty),
        "n_clusters": float(n_clusters),
        "n_non_singleton_clusters": float(n_non_singleton),
    }

def save_excel(out_xlsx: str, sheets: Dict[str, pd.DataFrame]):
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name)


# =========================
# Encoder (MLP AutoEncoder)
# =========================
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

def run_encoder_embedding(X_ra_log: np.ndarray, enc_seed: int, device: torch.device) -> np.ndarray:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_ra_log).astype(np.float32)

    set_global_seed(enc_seed)
    x = torch.from_numpy(X_scaled).to(device)

    model = MLPAutoEncoder(
        input_dim=x.shape[1],
        hidden_dim=ENC_HIDDEN_DIM,
        latent_dim=ENC_LATENT_DIM,
        dropout=ENC_DROPOUT
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=ENC_LR, weight_decay=ENC_WEIGHT_DECAY)

    model.train()
    for _ in range(int(ENC_EPOCHS)):
        opt.zero_grad(set_to_none=True)
        x_hat, _ = model(x)
        loss = F.mse_loss(x_hat, x)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        _, z = model(x)
    return z.detach().cpu().numpy().astype(np.float32)


# =========================
# GNN (GCN AutoEncoder)
# =========================
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        return torch.matmul(A_norm, h)

class GNNFeatureAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("GNN_NUM_LAYERS must be >= 1")
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
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

def train_gnn_autoencoder(features: np.ndarray, A_norm: np.ndarray, gnn_seed: int, device: torch.device) -> np.ndarray:
    set_global_seed(gnn_seed)

    x = torch.from_numpy(features.astype(np.float32)).to(device)
    A_t = torch.from_numpy(A_norm.astype(np.float32)).to(device)

    model = GNNFeatureAutoEncoder(in_dim=x.shape[1], hidden_dim=GNN_HIDDEN_DIM, num_layers=GNN_NUM_LAYERS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=GNN_LR)

    model.train()
    for _ in range(int(GNN_EPOCHS)):
        opt.zero_grad(set_to_none=True)
        x_hat, _ = model(x, A_t)
        loss = F.mse_loss(x_hat, x)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        _, z = model(x, A_t)
    return z.detach().cpu().numpy().astype(np.float32)


# =========================
# Interaction matrices
# =========================
def build_comp_compl_matrices(df_inter: pd.DataFrame, genomes: list) -> Tuple[np.ndarray, np.ndarray]:
    needed = {"Genome1", "Genome2", "Competition_Ave", "Complementarity_Ave"}
    missing = needed - set(df_inter.columns)
    if missing:
        raise ValueError(f"Interaction sheet 缺少列: {sorted(list(missing))}")

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

def compute_baseline_stats(comp_mat: np.ndarray, compl_mat: np.ndarray) -> Tuple[float, float]:
    mask = ~np.eye(comp_mat.shape[0], dtype=bool)
    comp_all = float(comp_mat[mask].mean()) if comp_mat[mask].size else 0.0
    compl_all = float(compl_mat[mask].mean()) if compl_mat[mask].size else 0.0
    return comp_all, compl_all


# =========================
# Main
# =========================
def main():
    if not isinstance(K, int) or K < 2:
        raise ValueError("K 必须是 >=2 的整数（Final 单次运行不扫 K）")

    device = get_device()
    print(f"[INFO] device={device}")
    print(f"[INFO] MODEL_TYPE={MODEL_TYPE}, EDGE_TYPE={EDGE_TYPE}, K={K}")
    print(f"[INFO] USE_BOOTSTRAP={USE_BOOTSTRAP}, BOOTSTRAP_B_IDX={BOOTSTRAP_B_IDX}")

    df_ra = pd.read_excel(INPUT_XLSX, sheet_name=RA_SHEET_NAME, index_col=0)
    df_inter = pd.read_excel(INPUT_XLSX, sheet_name=INTERACTION_SHEET_NAME)

    genomes = df_ra.index.astype(str).tolist()
    X_ra = df_ra.values.astype(np.float32)
    X_ra_log_full = np.log1p(X_ra)

    comp_mat_raw, compl_mat_raw = build_comp_compl_matrices(df_inter, genomes)
    comp_all, compl_all = compute_baseline_stats(comp_mat_raw, compl_mat_raw)

    if USE_BOOTSTRAP:
        n_samples = X_ra_log_full.shape[1]
        sample_seed_base = GLOBAL_RANDOM_SEED + 100000 * ENC_LATENT_DIM + 1000 * ENC_HIDDEN_DIM
        sample_indices = bootstrap_sample_indices(sample_seed_base, BOOTSTRAP_B_IDX, n_samples)
        X_ra_log = X_ra_log_full[:, sample_indices]
    else:
        X_ra_log = X_ra_log_full

    # ---- build A_norm ----
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
    A_norm = A_norm_dict[(MODEL_TYPE, EDGE_TYPE)]

    b = int(BOOTSTRAP_B_IDX) if USE_BOOTSTRAP else 0
    enc_seed = GLOBAL_RANDOM_SEED + 100000 * ENC_LATENT_DIM + 1000 * ENC_HIDDEN_DIM + b
    gnn_seed_base = (GLOBAL_RANDOM_SEED +
                     100000 * ENC_LATENT_DIM + 1000 * ENC_HIDDEN_DIM +
                     100 * GNN_HIDDEN_DIM + 10 * GNN_NUM_LAYERS)
    gnn_seed = gnn_seed_base + b + (0 if MODEL_TYPE == "Red" else 5000) + (0 if EDGE_TYPE == "single" else 7000)

    # ---- Encoder -> GNN ----
    Z_enc = run_encoder_embedding(X_ra_log, enc_seed=enc_seed, device=device)
    Z_gnn = train_gnn_autoencoder(Z_enc, A_norm=A_norm, gnn_seed=gnn_seed, device=device)

    # ---- KMeans： random_state=gnn_seed + K ----
    km = KMeans(
        n_clusters=K,
        n_init=int(KMEANS_N_INIT),
        max_iter=int(KMEANS_MAX_ITER),
        random_state=int(gnn_seed + K),
    )
    labels = km.fit_predict(Z_gnn).astype(int)

    metrics = evaluate_clustering(
        Z=Z_gnn,
        labels=labels,
        model_type=MODEL_TYPE,
        comp_mat=comp_mat_raw,
        compl_mat=compl_mat_raw,
        comp_all=comp_all,
        compl_all=compl_all,
    )
    print(f"[INFO] Sli={metrics['sli']:.4f} | Award={metrics['award']:.3f} | Penalty={metrics['penalty']:.3f} | Sli_AwardPenalty={metrics['sli_award_penalty']:.4f}")

    # ---- Labels ----
    df_labels = pd.DataFrame({"Genome": genomes, "Label": labels})
    df_labels["ClusterSize"] = df_labels["Label"].map(df_labels["Label"].value_counts())
    df_labels = df_labels.set_index("Genome").sort_values(["Label", "ClusterSize"], ascending=[True, False])

    # ---- PCA2D / TSNE2D  ----
    pca = PCA(n_components=2, random_state=GLOBAL_RANDOM_SEED)
    pca_xy = pca.fit_transform(Z_gnn)
    df_pca = pd.DataFrame(pca_xy, index=genomes, columns=["PC1", "PC2"])
    df_pca["Label"] = labels

    N = len(genomes)
    perplexity = min(30, max(5, (N - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=GLOBAL_RANDOM_SEED)
    tsne_xy = tsne.fit_transform(Z_gnn)
    df_tsne = pd.DataFrame(tsne_xy, index=genomes, columns=["tSNE1", "tSNE2"])
    df_tsne["Label"] = labels

    # ---- Summary CSV ----
    summary_row = {
        "MODEL_TYPE": MODEL_TYPE,
        "EDGE_TYPE": EDGE_TYPE,
        "K": K,
        "USE_BOOTSTRAP": USE_BOOTSTRAP,
        "BOOTSTRAP_B_IDX": BOOTSTRAP_B_IDX if USE_BOOTSTRAP else -1,
        "enc_seed": enc_seed,
        "gnn_seed": gnn_seed,
        "Sli": metrics["sli"],
        "Award": metrics["award"],
        "Penalty": metrics["penalty"],
        "Sli_Award": metrics["sli_award"],
        "Sli_AwardPenalty": metrics["sli_award_penalty"],
        "Comp_in": metrics["comp_in"],
        "Compl_in": metrics["compl_in"],
        "Comp_all": comp_all,
        "Compl_all": compl_all,
        "ENC_LATENT_DIM": ENC_LATENT_DIM,
        "ENC_HIDDEN_DIM": ENC_HIDDEN_DIM,
        "ENC_LR": ENC_LR,
        "ENC_EPOCHS": ENC_EPOCHS,
        "ENC_WEIGHT_DECAY": ENC_WEIGHT_DECAY,
        "ENC_DROPOUT": ENC_DROPOUT,
        "GNN_HIDDEN_DIM": GNN_HIDDEN_DIM,
        "GNN_NUM_LAYERS": GNN_NUM_LAYERS,
        "GNN_LR": GNN_LR,
        "GNN_EPOCHS": GNN_EPOCHS,
        "KMEANS_N_INIT": KMEANS_N_INIT,
        "KMEANS_MAX_ITER": KMEANS_MAX_ITER,
        "GLOBAL_RANDOM_SEED": GLOBAL_RANDOM_SEED,
        "TSNE_perplexity": perplexity,
    }
    pd.DataFrame([summary_row]).to_csv(os.path.join(OUT_DIR, "Final_ScoreSummary.csv"), index=False)

    # ---- Excel ----
    sheets = {
        "Labels": df_labels,
        "PCA2D": df_pca.sort_values("Label"),
        "TSNE2D": df_tsne.sort_values("Label"),
    }
    if SAVE_EMBEDDING:
        df_enc = pd.DataFrame(Z_enc, index=genomes, columns=[f"ENC_{i+1}" for i in range(Z_enc.shape[1])])
        df_enc["Label"] = labels
        df_gnn = pd.DataFrame(Z_gnn, index=genomes, columns=[f"GNN_{i+1}" for i in range(Z_gnn.shape[1])])
        df_gnn["Label"] = labels
        sheets["EncoderEmbedding"] = df_enc.sort_values("Label")
        sheets["GNNEmbedding"] = df_gnn.sort_values("Label")

    save_excel(os.path.join(OUT_DIR, "Final_MAG_Labels.xlsx"), sheets)

    print("[INFO] Done.")
    print(f"[INFO] Outputs: {OUT_DIR}")

if __name__ == "__main__":
    main()
