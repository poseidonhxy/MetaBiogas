#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ================== Setting ==================

# RA matrix
RA_XLSX = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/251211-Cluster/Encoder/Database.xlsx"
RA_SHEET_NAME = "RA"


INTERACTION_XLSX = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/251211-Cluster/Encoder/Database.xlsx"
INTERACTION_SHEET_NAME = "CCD"
OUT_DIR = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/251211-Cluster/Encoder/"

# ===== Hyperparameters=====
RA_LATENT_LIST = [1, 2, 3, 4, 5]
RA_HIDDEN_DIM_LIST = [16, 32, 64]
INTER_LATENT_LIST = [1, 2, 3]
INTER_HIDDEN_DIM_LIST = [16, 32, 64]
BATCH_SIZE_LIST = [32, 64]
K_RANGE = list(range(5, 11))
LR = 1e-3
N_EPOCHS_RA = 100
N_EPOCHS_INTER = 100
MIN_NON_SINGLETON_CLUSTERS = 5
N_BOOTSTRAP = 20
STAGE1_SLI_THRESHOLD = 0.4
GLOBAL_RANDOM_STATE = 20251213
KMEANS_N_INIT = 20
KMEANS_MAX_ITER = 500

EPS = 1e-8




def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_ra_data(xlsx_path: str, sheet_name: str):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if df.shape[1] < 2:
        raise ValueError("RA sheet must have at least 2 columns (ID + >=1 sample).")

    id_col = df.columns[0]
    mag_ids = df[id_col].astype(str).values
    sample_cols = df.columns[1:]

    X_raw = df[sample_cols].astype(float).values
    X_log = np.log1p(X_raw)

    print(f"[RA] read {df.shape[0]} MAG, {len(sample_cols)} samples.")
    return df, mag_ids, X_log, list(sample_cols)


def read_interaction_matrices(path: str, sheet_name, mag_ids):

    df = pd.read_excel(path, sheet_name=sheet_name)
    required_cols = {"Genome1", "Genome2", "Competition", "Complementarity"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Interaction sheet col: {required_cols}, input col: {df.columns.tolist()}")

    df["Genome1"] = df["Genome1"].astype(str)
    df["Genome2"] = df["Genome2"].astype(str)
    df = df[df["Genome1"].isin(mag_ids) & df["Genome2"].isin(mag_ids)].copy()

    def build_matrix(value_col: str) -> pd.DataFrame:
        sub = df[["Genome1", "Genome2", value_col]].copy()
        mat = sub.pivot(index="Genome1", columns="Genome2", values=value_col)
        mat = mat.reindex(index=mag_ids, columns=mag_ids)
        mat = mat.fillna(0.0)
        for g in mag_ids:
            mat.loc[g, g] = 0.0
        return mat

    comp_mat = build_matrix("Competition")
    compl_mat = build_matrix("Complementarity")

    print("[Inter] Competition dimension:", comp_mat.shape)
    print("[Inter] Complementarity dimension:", compl_mat.shape)
    return comp_mat, compl_mat


def build_block_raw(mat: pd.DataFrame, mag_ids):

    rows = []
    for g in mag_ids:
        vals = []
        row = mat.loc[g]
        for h in mag_ids:
            if h == g:
                continue
            vals.append(row[h])
        rows.append(vals)
    return np.array(rows, dtype=float)


def compute_baseline_from_matrices(comp_mat: pd.DataFrame, compl_mat: pd.DataFrame):

    comp_vals = comp_mat.values.astype(float)
    compl_vals = compl_mat.values.astype(float)
    n = comp_vals.shape[0]
    mask_offdiag = ~np.eye(n, dtype=bool)

    comp_all = comp_vals[mask_offdiag].mean()
    compl_all = compl_vals[mask_offdiag].mean()

    print(f"[Baseline] comp_all={comp_all:.4f}, compl_all={compl_all:.4f}")
    return comp_all, compl_all


def compute_cluster_interactions(comp_mat: pd.DataFrame,
                                 compl_mat: pd.DataFrame,
                                 mag_ids,
                                 labels):

    mag_ids = np.asarray(mag_ids)
    labels = np.asarray(labels)
    n_mag = len(mag_ids)

    comp_vals = []
    compl_vals = []

    for idx_i in range(n_mag):
        gi = mag_ids[idx_i]
        ci = labels[idx_i]
        row_comp = comp_mat.loc[gi]
        row_compl = compl_mat.loc[gi]
        for idx_j in range(n_mag):
            if idx_j == idx_i:
                continue
            if labels[idx_j] != ci:
                continue
            gj = mag_ids[idx_j]
            comp_vals.append(row_comp[gj])
            compl_vals.append(row_compl[gj])

    n_edges = len(comp_vals)
    if n_edges == 0:
        return np.nan, np.nan, 0

    comp_in = float(np.mean(comp_vals))
    compl_in = float(np.mean(compl_vals))
    return comp_in, compl_in, n_edges


# ================== AutoEncoder  ==================


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(
    X: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    lr: float,
    batch_size: int,
    n_epochs: int,
    device: torch.device,
    rng: np.random.RandomState,
):

    seed = int(rng.randint(0, 10**9))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    N, D = X.shape
    assert D == input_dim, f"input dimension unmatch: X.shape={X.shape}, input_dim={input_dim}"

    X_tensor = torch.from_numpy(X.astype(np.float32))
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=min(batch_size, N), shuffle=True)

    model = AutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        for batch in loader:
            xb = batch[0].to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        Z = model.encode(X_tensor.to(device)).cpu().numpy()

    return model, Z


# ================== clustering evluation ==================


def evaluate_clustering(
    X_feat: np.ndarray,
    labels: np.ndarray,
    comp_mat: pd.DataFrame,
    compl_mat: pd.DataFrame,
    mag_ids,
    comp_all: float,
    compl_all: float,
    model_type: str,
):

    unique_labels = np.unique(labels)
    sizes = np.bincount(labels, minlength=unique_labels.size)

    # 结构指标
    if unique_labels.size < 2:
        sil = np.nan
        ch = np.nan
        db = np.nan
    else:
        try:
            sil = float(silhouette_score(X_feat, labels))
        except Exception:
            sil = np.nan
        try:
            ch = float(calinski_harabasz_score(X_feat, labels))
        except Exception:
            ch = np.nan
        try:
            db = float(davies_bouldin_score(X_feat, labels))
        except Exception:
            db = np.nan

    n_singleton = int(np.sum(sizes == 1))
    n_non_singleton = int(np.sum(sizes > 1))
    min_size_non_singleton = int(sizes[sizes > 1].min()) if n_non_singleton > 0 else 0
    max_size = int(sizes.max()) if sizes.size > 0 else 0
    valid_structure = (unique_labels.size >= 2) and (n_non_singleton >= MIN_NON_SINGLETON_CLUSTERS)
    if valid_structure and not np.isnan(sil):
        comp_in, compl_in, n_edges = compute_cluster_interactions(
            comp_mat=comp_mat,
            compl_mat=compl_mat,
            mag_ids=mag_ids,
            labels=labels,
        )
        if n_edges == 0 or np.isnan(comp_in) or np.isnan(compl_in):
            Sli_only = np.nan
            Sli_Award = np.nan
            Sli_AwardPenalty = np.nan
        else:
            Sli_only = sil
            if model_type == "Red":
                Award = comp_in / (comp_all + EPS)
                ratio_compl = (compl_in + EPS) / (compl_all + EPS)
                Penalty = 1.0 if ratio_compl <= 1.0 else (compl_all + EPS) / (compl_in + EPS)
            elif model_type == "Sup":
                Award = compl_in / (compl_all + EPS)
                ratio_comp = (comp_in + EPS) / (comp_all + EPS)
                Penalty = 1.0 if ratio_comp <= 1.0 else (comp_all + EPS) / (comp_in + EPS)
            else:
                raise ValueError(f"unknown model_type: {model_type}")

            Sli_Award = Sli_only * Award
            Sli_AwardPenalty = Sli_Award * Penalty
    else:
        comp_in = np.nan
        compl_in = np.nan
        n_edges = 0
        Sli_only = np.nan
        Sli_Award = np.nan
        Sli_AwardPenalty = np.nan

    result = {
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "n_singleton_clusters": n_singleton,
        "n_non_singleton_clusters": n_non_singleton,
        "min_cluster_size_non_singleton": min_size_non_singleton,
        "max_cluster_size": max_size,
        "valid_structure": valid_structure,
        "comp_in": comp_in,
        "compl_in": compl_in,
        "n_edges_within": n_edges,
        "Sli_only": Sli_only,
        "Sli_Award": Sli_Award,
        "Sli_AwardPenalty": Sli_AwardPenalty,
    }
    return result




def run_stage0(
    device,
    rng_global,
    X_ra_log,
    comp_raw,
    compl_raw,
    comp_mat,
    compl_mat,
    mag_ids,
    comp_all,
    compl_all,
):

    n_mag, n_samples = X_ra_log.shape
    _, n_inter_feat = comp_raw.shape

    print(
        f"[Stage0] n_mag={n_mag}, n_samples={n_samples}, "
        f"comp_feat={comp_raw.shape[1]}, compl_feat={compl_raw.shape[1]}"
    )

    stage0_records = []

    hyperparam_combos = []
    for ra_latent in RA_LATENT_LIST:
        for ra_hidden in RA_HIDDEN_DIM_LIST:
            for inter_latent in INTER_LATENT_LIST:
                for inter_hidden in INTER_HIDDEN_DIM_LIST:
                    for batch_size in BATCH_SIZE_LIST:
                        for K in K_RANGE:
                            hyperparam_combos.append(
                                (ra_latent, ra_hidden, inter_latent, inter_hidden, batch_size, K)
                            )


    for idx, combo in enumerate(hyperparam_combos, start=1):
        ra_latent, ra_hidden, inter_latent, inter_hidden, batch_size, K = combo
        print(
            f"[Stage0] Combo {idx}/{len(hyperparam_combos)}: "
            f"RA_latent={ra_latent}, RA_hidden={ra_hidden}, "
            f"Inter_latent={inter_latent}, Inter_hidden={inter_hidden}, "
            f"batch={batch_size}, K={K}"
        )

        rng = np.random.RandomState(rng_global.randint(0, 10**9))

        try:
            _, Z_ra = train_autoencoder(
                X=X_ra_log,
                input_dim=n_samples,
                hidden_dim=ra_hidden,
                latent_dim=ra_latent,
                lr=LR,
                batch_size=batch_size,
                n_epochs=N_EPOCHS_RA,
                device=device,
                rng=rng,
            )
        except Exception as e:
            print(f"[WARN]  RA Encoder training failed skip combine: {e}")
            for model_type in ["Red", "Sup"]:
                stage0_records.append({
                    "model_type": model_type,
                    "RA_latent": ra_latent,
                    "RA_hidden": ra_hidden,
                    "Inter_latent": inter_latent,
                    "Inter_hidden": inter_hidden,
                    "batch_size": batch_size,
                    "K": K,
                    "Sli_only": np.nan,
                    "Sli_Award": np.nan,
                    "Sli_AwardPenalty": np.nan,
                    "silhouette": np.nan,
                    "calinski_harabasz": np.nan,
                    "davies_bouldin": np.nan,
                    "n_singleton_clusters": 0,
                    "n_non_singleton_clusters": 0,
                    "min_cluster_size_non_singleton": 0,
                    "max_cluster_size": 0,
                    "valid_structure": False,
                    "comp_in": np.nan,
                    "compl_in": np.nan,
                    "n_edges_within": 0,
                })
            continue

        # 2) Competition Encoder
        try:
            _, Z_comp = train_autoencoder(
                X=comp_raw,
                input_dim=n_inter_feat,
                hidden_dim=inter_hidden,
                latent_dim=inter_latent,
                lr=LR,
                batch_size=batch_size,
                n_epochs=N_EPOCHS_INTER,
                device=device,
                rng=rng,
            )
        except Exception as e:
            print(f"[WARN]  Competition Encoder failed（Red invalid）: {e}")
            Z_comp = None

        # 3) Complementarity Encoder
        try:
            _, Z_compl = train_autoencoder(
                X=compl_raw,
                input_dim=n_inter_feat,
                hidden_dim=inter_hidden,
                latent_dim=inter_latent,
                lr=LR,
                batch_size=batch_size,
                n_epochs=N_EPOCHS_INTER,
                device=device,
                rng=rng,
            )
        except Exception as e:
            print(f"[WARN]  Complementarity Encoder failed（Sup invalid）: {e}")
            Z_compl = None

        # ---- Red  ----
        if Z_comp is not None:
            X_red = np.concatenate([Z_ra, Z_comp], axis=1)
            X_red_std = StandardScaler().fit_transform(X_red)
            kmeans_red = KMeans(
                n_clusters=K,
                random_state=int(rng.randint(0, 10**9)),
                n_init=KMEANS_N_INIT,
                max_iter=KMEANS_MAX_ITER,
            )
            labels_red = kmeans_red.fit_predict(X_red_std)

            eval_red = evaluate_clustering(
                X_feat=X_red_std,
                labels=labels_red,
                comp_mat=comp_mat,
                compl_mat=compl_mat,
                mag_ids=mag_ids,
                comp_all=comp_all,
                compl_all=compl_all,
                model_type="Red",
            )
        else:
            eval_red = {
                "Sli_only": np.nan,
                "Sli_Award": np.nan,
                "Sli_AwardPenalty": np.nan,
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "n_singleton_clusters": 0,
                "n_non_singleton_clusters": 0,
                "min_cluster_size_non_singleton": 0,
                "max_cluster_size": 0,
                "valid_structure": False,
                "comp_in": np.nan,
                "compl_in": np.nan,
                "n_edges_within": 0,
            }

        stage0_records.append({
            "model_type": "Red",
            "RA_latent": ra_latent,
            "RA_hidden": ra_hidden,
            "Inter_latent": inter_latent,
            "Inter_hidden": inter_hidden,
            "batch_size": batch_size,
            "K": K,
            "Sli_only": eval_red["Sli_only"],
            "Sli_Award": eval_red["Sli_Award"],
            "Sli_AwardPenalty": eval_red["Sli_AwardPenalty"],
            "silhouette": eval_red["silhouette"],
            "calinski_harabasz": eval_red["calinski_harabasz"],
            "davies_bouldin": eval_red["davies_bouldin"],
            "n_singleton_clusters": eval_red["n_singleton_clusters"],
            "n_non_singleton_clusters": eval_red["n_non_singleton_clusters"],
            "min_cluster_size_non_singleton": eval_red["min_cluster_size_non_singleton"],
            "max_cluster_size": eval_red["max_cluster_size"],
            "valid_structure": eval_red["valid_structure"],
            "comp_in": eval_red["comp_in"],
            "compl_in": eval_red["compl_in"],
            "n_edges_within": eval_red["n_edges_within"],
        })

        # ---- Sup  ----
        if Z_compl is not None:
            X_sup = np.concatenate([Z_ra, Z_compl], axis=1)
            X_sup_std = StandardScaler().fit_transform(X_sup)
            kmeans_sup = KMeans(
                n_clusters=K,
                random_state=int(rng.randint(0, 10**9)),
                n_init=KMEANS_N_INIT,
                max_iter=KMEANS_MAX_ITER,
            )
            labels_sup = kmeans_sup.fit_predict(X_sup_std)

            eval_sup = evaluate_clustering(
                X_feat=X_sup_std,
                labels=labels_sup,
                comp_mat=comp_mat,
                compl_mat=compl_mat,
                mag_ids=mag_ids,
                comp_all=comp_all,
                compl_all=compl_all,
                model_type="Sup",
            )
        else:
            eval_sup = {
                "Sli_only": np.nan,
                "Sli_Award": np.nan,
                "Sli_AwardPenalty": np.nan,
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "n_singleton_clusters": 0,
                "n_non_singleton_clusters": 0,
                "min_cluster_size_non_singleton": 0,
                "max_cluster_size": 0,
                "valid_structure": False,
                "comp_in": np.nan,
                "compl_in": np.nan,
                "n_edges_within": 0,
            }

        stage0_records.append({
            "model_type": "Sup",
            "RA_latent": ra_latent,
            "RA_hidden": ra_hidden,
            "Inter_latent": inter_latent,
            "Inter_hidden": inter_hidden,
            "batch_size": batch_size,
            "K": K,
            "Sli_only": eval_sup["Sli_only"],
            "Sli_Award": eval_sup["Sli_Award"],
            "Sli_AwardPenalty": eval_sup["Sli_AwardPenalty"],
            "silhouette": eval_sup["silhouette"],
            "calinski_harabasz": eval_sup["calinski_harabasz"],
            "davies_bouldin": eval_sup["davies_bouldin"],
            "n_singleton_clusters": eval_sup["n_singleton_clusters"],
            "n_non_singleton_clusters": eval_sup["n_non_singleton_clusters"],
            "min_cluster_size_non_singleton": eval_sup["min_cluster_size_non_singleton"],
            "max_cluster_size": eval_sup["max_cluster_size"],
            "valid_structure": eval_sup["valid_structure"],
            "comp_in": eval_sup["comp_in"],
            "compl_in": eval_sup["compl_in"],
            "n_edges_within": eval_sup["n_edges_within"],
        })

    df_stage0 = pd.DataFrame(stage0_records)
    return df_stage0


# ================== Bootstrap for Sli>=0.4 combine ==================


def run_stage1_bootstrap(
    device,
    rng_global,
    X_ra_log,
    comp_raw,
    compl_raw,
    comp_mat,
    compl_mat,
    mag_ids,
    comp_all,
    compl_all,
    df_stage0,
):

    n_mag, n_samples = X_ra_log.shape
    _, n_inter_feat = comp_raw.shape

    mask = (
        df_stage0["valid_structure"].astype(bool)
        & df_stage0["Sli_only"].notna()
        & (df_stage0["Sli_only"] >= STAGE1_SLI_THRESHOLD)
    )
    df_candidates = df_stage0.loc[mask, [
        "model_type",
        "RA_latent",
        "RA_hidden",
        "Inter_latent",
        "Inter_hidden",
        "batch_size",
        "K",
    ]].drop_duplicates().reset_index(drop=True)

    if df_candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    print(df_candidates)

    boot_records = []

    for idx, row in enumerate(df_candidates.itertuples(index=False), start=1):
        model_type = row.model_type  # "Red" or "Sup"
        ra_latent = int(row.RA_latent)
        ra_hidden = int(row.RA_hidden)
        inter_latent = int(row.Inter_latent)
        inter_hidden = int(row.Inter_hidden)
        batch_size = int(row.batch_size)
        K = int(row.K)

        print(
            f"\n[Stage1 Combo {idx}/{len(df_candidates)}] "
            f"model_type={model_type}, RA_latent={ra_latent}, RA_hidden={ra_hidden}, "
            f"Inter_latent={inter_latent}, Inter_hidden={inter_hidden}, "
            f"batch_size={batch_size}, K={K}"
        )

        for b in range(1, N_BOOTSTRAP + 1):
            rng = np.random.RandomState(rng_global.randint(0, 10**9))

            # ---- RA bootstrap: sample  ----
            ra_idx_boot = rng.randint(0, n_samples, size=n_samples)
            X_ra_boot = X_ra_log[:, ra_idx_boot]

            try:
                _, Z_ra = train_autoencoder(
                    X=X_ra_boot,
                    input_dim=n_samples,
                    hidden_dim=ra_hidden,
                    latent_dim=ra_latent,
                    lr=LR,
                    batch_size=batch_size,
                    n_epochs=N_EPOCHS_RA,
                    device=device,
                    rng=rng,
                )
            except Exception as e:
                print(f"[WARN] Stage1 RA Encoder bootstrap failed (b={b}): {e}")
                boot_records.append({
                    "model_type": model_type,
                    "RA_latent": ra_latent,
                    "RA_hidden": ra_hidden,
                    "Inter_latent": inter_latent,
                    "Inter_hidden": inter_hidden,
                    "batch_size": batch_size,
                    "K": K,
                    "bootstrap_id": b,
                    "Sli_only": np.nan,
                    "Sli_Award": np.nan,
                    "Sli_AwardPenalty": np.nan,
                    "silhouette": np.nan,
                    "calinski_harabasz": np.nan,
                    "davies_bouldin": np.nan,
                    "n_singleton_clusters": 0,
                    "n_non_singleton_clusters": 0,
                    "min_cluster_size_non_singleton": 0,
                    "max_cluster_size": 0,
                    "valid_structure": False,
                    "comp_in": np.nan,
                    "compl_in": np.nan,
                    "n_edges_within": 0,
                })
                continue

            # ---- Interaction bootstrap: feature sampling ----
            if model_type == "Red":
                inter_raw = comp_raw
            elif model_type == "Sup":
                inter_raw = compl_raw
            else:
                raise ValueError(f"unknown model_type: {model_type}")

            inter_idx_boot = rng.randint(0, n_inter_feat, size=n_inter_feat)
            inter_boot = inter_raw[:, inter_idx_boot]

            try:
                _, Z_inter = train_autoencoder(
                    X=inter_boot,
                    input_dim=n_inter_feat,
                    hidden_dim=inter_hidden,
                    latent_dim=inter_latent,
                    lr=LR,
                    batch_size=batch_size,
                    n_epochs=N_EPOCHS_INTER,
                    device=device,
                    rng=rng,
                )
            except Exception as e:
                print(f"[WARN] Stage1 Interaction Encoder bootstrap failed (b={b}): {e}")
                boot_records.append({
                    "model_type": model_type,
                    "RA_latent": ra_latent,
                    "RA_hidden": ra_hidden,
                    "Inter_latent": inter_latent,
                    "Inter_hidden": inter_hidden,
                    "batch_size": batch_size,
                    "K": K,
                    "bootstrap_id": b,
                    "Sli_only": np.nan,
                    "Sli_Award": np.nan,
                    "Sli_AwardPenalty": np.nan,
                    "silhouette": np.nan,
                    "calinski_harabasz": np.nan,
                    "davies_bouldin": np.nan,
                    "n_singleton_clusters": 0,
                    "n_non_singleton_clusters": 0,
                    "min_cluster_size_non_singleton": 0,
                    "max_cluster_size": 0,
                    "valid_structure": False,
                    "comp_in": np.nan,
                    "compl_in": np.nan,
                    "n_edges_within": 0,
                })
                continue

            # ----  latent + KMeans ----
            X_feat = np.concatenate([Z_ra, Z_inter], axis=1)
            X_feat_std = StandardScaler().fit_transform(X_feat)

            kmeans = KMeans(
                n_clusters=K,
                random_state=int(rng.randint(0, 10**9)),
                n_init=KMEANS_N_INIT,
                max_iter=KMEANS_MAX_ITER,
            )
            labels = kmeans.fit_predict(X_feat_std)

            eval_res = evaluate_clustering(
                X_feat=X_feat_std,
                labels=labels,
                comp_mat=comp_mat,
                compl_mat=compl_mat,
                mag_ids=mag_ids,
                comp_all=comp_all,
                compl_all=compl_all,
                model_type=model_type,
            )

            boot_records.append({
                "model_type": model_type,
                "RA_latent": ra_latent,
                "RA_hidden": ra_hidden,
                "Inter_latent": inter_latent,
                "Inter_hidden": inter_hidden,
                "batch_size": batch_size,
                "K": K,
                "bootstrap_id": b,
                "Sli_only": eval_res["Sli_only"],
                "Sli_Award": eval_res["Sli_Award"],
                "Sli_AwardPenalty": eval_res["Sli_AwardPenalty"],
                "silhouette": eval_res["silhouette"],
                "calinski_harabasz": eval_res["calinski_harabasz"],
                "davies_bouldin": eval_res["davies_bouldin"],
                "n_singleton_clusters": eval_res["n_singleton_clusters"],
                "n_non_singleton_clusters": eval_res["n_non_singleton_clusters"],
                "min_cluster_size_non_singleton": eval_res["min_cluster_size_non_singleton"],
                "max_cluster_size": eval_res["max_cluster_size"],
                "valid_structure": eval_res["valid_structure"],
                "comp_in": eval_res["comp_in"],
                "compl_in": eval_res["compl_in"],
                "n_edges_within": eval_res["n_edges_within"],
            })

    df_boot = pd.DataFrame(boot_records)

    # summary
    summary_records = []
    group_cols = [
        "model_type",
        "RA_latent",
        "RA_hidden",
        "Inter_latent",
        "Inter_hidden",
        "batch_size",
        "K",
    ]

    for combo_key, grp in df_boot.groupby(group_cols):
        row = dict(zip(group_cols, combo_key))
        row["n_bootstrap_total"] = len(grp)

        grp_valid = grp[grp["valid_structure"].astype(bool) & grp["Sli_only"].notna()]
        row["n_valid_bootstrap"] = len(grp_valid)

        if len(grp_valid) > 0:
            row["Sli_only_mean"] = grp_valid["Sli_only"].mean()
            row["Sli_only_std"] = grp_valid["Sli_only"].std(ddof=1)
            row["Sli_Award_mean"] = grp_valid["Sli_Award"].mean()
            row["Sli_AwardPenalty_mean"] = grp_valid["Sli_AwardPenalty"].mean()
            row["comp_in_mean"] = grp_valid["comp_in"].mean()
            row["compl_in_mean"] = grp_valid["compl_in"].mean()
            row["silhouette_mean"] = grp_valid["silhouette"].mean()
            row["calinski_harabasz_mean"] = grp_valid["calinski_harabasz"].mean()
            row["davies_bouldin_mean"] = grp_valid["davies_bouldin"].mean()
            row["n_non_singleton_clusters_mean"] = grp_valid["n_non_singleton_clusters"].mean()
        else:
            row["Sli_only_mean"] = np.nan
            row["Sli_only_std"] = np.nan
            row["Sli_Award_mean"] = np.nan
            row["Sli_AwardPenalty_mean"] = np.nan
            row["comp_in_mean"] = np.nan
            row["compl_in_mean"] = np.nan
            row["silhouette_mean"] = np.nan
            row["calinski_harabasz_mean"] = np.nan
            row["davies_bouldin_mean"] = np.nan
            row["n_non_singleton_clusters_mean"] = np.nan

        summary_records.append(row)

    df_summary = pd.DataFrame(summary_records)
    return df_boot, df_summary


# ================== Main ==================


def main():
    ensure_outdir(OUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rng_global = np.random.RandomState(GLOBAL_RANDOM_STATE)

    #  RA + Interaction input
    df_ra, mag_ids, X_ra_log, sample_names = load_ra_data(RA_XLSX, RA_SHEET_NAME)
    comp_mat, compl_mat = read_interaction_matrices(INTERACTION_XLSX, INTERACTION_SHEET_NAME, mag_ids)
    comp_all, compl_all = compute_baseline_from_matrices(comp_mat, compl_mat)

    comp_raw = build_block_raw(comp_mat, mag_ids)
    compl_raw = build_block_raw(compl_mat, mag_ids)

    # Hypermeters tuning
    print("\n========== Stage0: filtering without bootstrap ==========")
    df_stage0 = run_stage0(
        device=device,
        rng_global=rng_global,
        X_ra_log=X_ra_log,
        comp_raw=comp_raw,
        compl_raw=compl_raw,
        comp_mat=comp_mat,
        compl_mat=compl_mat,
        mag_ids=mag_ids,
        comp_all=comp_all,
        compl_all=compl_all,
    )
    stage0_csv = os.path.join(OUT_DIR, "Encoder_RedSup_Stage0_Small.csv")
    df_stage0.to_csv(stage0_csv, index=False)

    # bootstrap
    df_boot, df_summary = run_stage1_bootstrap(
        device=device,
        rng_global=rng_global,
        X_ra_log=X_ra_log,
        comp_raw=comp_raw,
        compl_raw=compl_raw,
        comp_mat=comp_mat,
        compl_mat=compl_mat,
        mag_ids=mag_ids,
        comp_all=comp_all,
        compl_all=compl_all,
        df_stage0=df_stage0,
    )

    if not df_boot.empty:
        boot_csv = os.path.join(OUT_DIR, "Encoder_RedSup_Bootstrap_Raw.csv")
        df_boot.to_csv(boot_csv, index=False)

    if not df_summary.empty:
        summary_csv = os.path.join(OUT_DIR, "Encoder_RedSup_Bootstrap_Summary_Sli04_20boot.csv")
        df_summary.to_csv(summary_csv, index=False)


if __name__ == "__main__":
    main()

