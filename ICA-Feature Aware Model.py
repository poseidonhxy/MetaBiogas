#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# ================== config ==================

# RA matrix
RA_XLSX = "/lustre/home/acct-clsbfw/zx2018/HXY/MetaBiogas/summary/Analysis/251211_cluster/ICA/Database.xlsx"
RA_SHEET_NAME = "RA"
INTERACTION_XLSX = "/lustre/home/acct-clsbfw/zx2018/HXY/MetaBiogas/summary/Analysis/251211_cluster/ICA/Database.xlsx"
INTERACTION_SHEET_NAME = "CCD"
OUT_DIR = "/lustre/home/acct-clsbfw/zx2018/HXY/MetaBiogas/summary/Analysis/251211_cluster/ICA"

RA_N_COMPONENTS_LIST = list(range(1, 11))

INTER_N_COMPONENTS_MAX = 5
K_RANGE = list(range(5, 21))

N_BOOTSTRAP = 50
MIN_NON_SINGLETON_CLUSTER_SIZE = 5
GLOBAL_RANDOM_STATE = 20251211

RA_ICA_MAX_ITER = 2000
INTER_ICA_MAX_ITER = 2000

KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 1000

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
    # log1p
    X_log = np.log1p(X_raw)

    print(f"[RA] 读入 {df.shape[0]} MAG, {len(sample_cols)} samples.")
    return df, mag_ids, X_log, list(sample_cols)


def read_interaction_matrices(path: str, sheet_name, mag_ids):

    df = pd.read_excel(path, sheet_name=sheet_name)
    required_cols = {"Genome1", "Genome2", "Competition", "Complementarity"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Interaction sheet require col: {required_cols}, col: {df.columns.tolist()}")

    df["Genome1"] = df["Genome1"].astype(str)
    df["Genome2"] = df["Genome2"].astype(str)
    df = df[df["Genome1"].isin(mag_ids) & df["Genome2"].isin(mag_ids)].copy()
    if df.empty:
        raise ValueError("Interaction MAG inmatch")

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

    print("[Inter] Competition matrix:", comp_mat.shape)
    print("[Inter] Complementarity matrix:", compl_mat.shape)
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


def run_ica_block(
    X_block: np.ndarray,
    target_n_components: int,
    random_state: int,
    max_iter: int,
    max_n_for_this_block: int = None,
):

    n_mag, n_features = X_block.shape

    max_allowed_by_shape = min(n_mag, n_features)
    if max_n_for_this_block is not None:
        max_allowed_by_shape = min(max_allowed_by_shape, max_n_for_this_block)

    n_components = min(target_n_components, max_allowed_by_shape)
    if n_components < 1:
        raise ValueError("n_components = 0")

    ica = FastICA(
        n_components=n_components,
        random_state=random_state,
        max_iter=max_iter,
        whiten="unit-variance",
    )
    X_ica = ica.fit_transform(X_block)
    return X_ica, n_components


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


def main():
    ensure_outdir(OUT_DIR)

    rng_global = np.random.RandomState(GLOBAL_RANDOM_STATE)
    df_ra, mag_ids, X_ra_log, sample_names = load_ra_data(RA_XLSX, RA_SHEET_NAME)
    comp_mat, compl_mat = read_interaction_matrices(INTERACTION_XLSX, INTERACTION_SHEET_NAME, mag_ids)
    comp_all, compl_all = compute_baseline_from_matrices(comp_mat, compl_mat)
    comp_raw = build_block_raw(comp_mat, mag_ids)
    compl_raw = build_block_raw(compl_mat, mag_ids)

    n_mag, n_samples = X_ra_log.shape
    print(
        f"[Info] n_mag={n_mag}, n_samples={n_samples}, "
        f"comp_feat={comp_raw.shape[1]}, compl_feat={compl_raw.shape[1]}"
    )

    records = []

    for n_components in RA_N_COMPONENTS_LIST:
        for k in K_RANGE:
            print(f"\n[Hyper] n_components={n_components}, K={k}")
            for b in range(N_BOOTSTRAP):
                rng = np.random.RandomState(rng_global.randint(0, 10**9))

                ra_idx_boot = rng.randint(0, n_samples, size=n_samples)
                X_ra_boot = X_ra_log[:, ra_idx_boot]
                X_ra_ica, n_ra_used = run_ica_block(
                    X_ra_boot,
                    target_n_components=n_components,
                    random_state=rng.randint(0, 10**9),
                    max_iter=RA_ICA_MAX_ITER,
                    max_n_for_this_block=None,
                )

                _, n_comp_feat = comp_raw.shape
                comp_idx_boot = rng.randint(0, n_comp_feat, size=n_comp_feat)
                comp_boot = comp_raw[:, comp_idx_boot]
                X_comp_ica, n_comp_used = run_ica_block(
                    comp_boot,
                    target_n_components=n_components,
                    random_state=rng.randint(0, 10**9),
                    max_iter=INTER_ICA_MAX_ITER,
                    max_n_for_this_block=INTER_N_COMPONENTS_MAX,
                )

                X_red = np.concatenate([X_ra_ica, X_comp_ica], axis=1)
                X_red_std = StandardScaler().fit_transform(X_red)

                kmeans_red = KMeans(
                    n_clusters=k,
                    random_state=rng.randint(0, 10**9),
                    n_init=KMEANS_N_INIT,
                    max_iter=KMEANS_MAX_ITER,
                )
                labels_red = kmeans_red.fit_predict(X_red_std)

                unique_red = np.unique(labels_red)
                if unique_red.size < 2:
                    sil_red = np.nan
                    ch_red = np.nan
                    db_red = np.nan
                    sizes_red = np.bincount(labels_red, minlength=k)
                else:
                    sil_red = float(silhouette_score(X_red_std, labels_red))
                    ch_red = float(calinski_harabasz_score(X_red_std, labels_red))
                    db_red = float(davies_bouldin_score(X_red_std, labels_red))
                    sizes_red = np.bincount(labels_red, minlength=k)

                n_singleton_red = int(np.sum(sizes_red == 1))
                n_non_singleton_red = int(np.sum(sizes_red > 1))
                if n_non_singleton_red > 0:
                    min_size_non_singleton_red = int(sizes_red[sizes_red > 1].min())
                else:
                    min_size_non_singleton_red = 0
                max_size_red = int(sizes_red.max())

                valid_structure_red = (
                    (unique_red.size >= 2)
                    and (n_non_singleton_red > 0)
                    and (min_size_non_singleton_red >= MIN_NON_SINGLETON_CLUSTER_SIZE)
                )

                if valid_structure_red and not np.isnan(sil_red):
                    comp_in_red, compl_in_red, n_edges_red = compute_cluster_interactions(
                        comp_mat=comp_mat,
                        compl_mat=compl_mat,
                        mag_ids=mag_ids,
                        labels=labels_red,
                    )
                    if n_edges_red == 0 or np.isnan(comp_in_red) or np.isnan(compl_in_red):
                        Red_Sli_only = np.nan
                        Red_Sli_Award = np.nan
                        Red_Sli_AwardPenalty = np.nan
                    else:
                        Award_Red = comp_in_red / (comp_all + EPS)
                        ratio_compl = (compl_in_red + EPS) / (compl_all + EPS)
                        Penalty_Red = 1.0 if ratio_compl <= 1.0 else (compl_all + EPS) / (compl_in_red + EPS)
                        Red_Sli_only = sil_red
                        Red_Sli_Award = Red_Sli_only * Award_Red
                        Red_Sli_AwardPenalty = Red_Sli_Award * Penalty_Red
                else:
                    comp_in_red = np.nan
                    compl_in_red = np.nan
                    n_edges_red = 0
                    Red_Sli_only = np.nan
                    Red_Sli_Award = np.nan
                    Red_Sli_AwardPenalty = np.nan

                # ============= Sup  =============
                _, n_compl_feat = compl_raw.shape
                compl_idx_boot = rng.randint(0, n_compl_feat, size=n_compl_feat)
                compl_boot = compl_raw[:, compl_idx_boot]
                X_compl_ica, n_compl_used = run_ica_block(
                    compl_boot,
                    target_n_components=n_components,
                    random_state=rng.randint(0, 10**9),
                    max_iter=INTER_ICA_MAX_ITER,
                    max_n_for_this_block=INTER_N_COMPONENTS_MAX,  # Complementarity ICA 维度上限 5
                )

                X_sup = np.concatenate([X_ra_ica, X_compl_ica], axis=1)
                X_sup_std = StandardScaler().fit_transform(X_sup)

                kmeans_sup = KMeans(
                    n_clusters=k,
                    random_state=rng.randint(0, 10**9),
                    n_init=KMEANS_N_INIT,
                    max_iter=KMEANS_MAX_ITER,
                )
                labels_sup = kmeans_sup.fit_predict(X_sup_std)

                unique_sup = np.unique(labels_sup)
                if unique_sup.size < 2:
                    sil_sup = np.nan
                    ch_sup = np.nan
                    db_sup = np.nan
                    sizes_sup = np.bincount(labels_sup, minlength=k)
                else:
                    sil_sup = float(silhouette_score(X_sup_std, labels_sup))
                    ch_sup = float(calinski_harabasz_score(X_sup_std, labels_sup))
                    db_sup = float(davies_bouldin_score(X_sup_std, labels_sup))
                    sizes_sup = np.bincount(labels_sup, minlength=k)

                n_singleton_sup = int(np.sum(sizes_sup == 1))
                n_non_singleton_sup = int(np.sum(sizes_sup > 1))
                if n_non_singleton_sup > 0:
                    min_size_non_singleton_sup = int(sizes_sup[sizes_sup > 1].min())
                else:
                    min_size_non_singleton_sup = 0
                max_size_sup = int(sizes_sup.max())

                valid_structure_sup = (
                    (unique_sup.size >= 2)
                    and (n_non_singleton_sup > 0)
                    and (min_size_non_singleton_sup >= MIN_NON_SINGLETON_CLUSTER_SIZE)
                )

                if valid_structure_sup and not np.isnan(sil_sup):
                    comp_in_sup, compl_in_sup, n_edges_sup = compute_cluster_interactions(
                        comp_mat=comp_mat,
                        compl_mat=compl_mat,
                        mag_ids=mag_ids,
                        labels=labels_sup,
                    )
                    if n_edges_sup == 0 or np.isnan(comp_in_sup) or np.isnan(compl_in_sup):
                        Sup_Sli_only = np.nan
                        Sup_Sli_Award = np.nan
                        Sup_Sli_AwardPenalty = np.nan
                    else:
                        Award_Sup = compl_in_sup / (compl_all + EPS)
                        ratio_comp = (comp_in_sup + EPS) / (comp_all + EPS)
                        # 只惩罚竞争高于 baseline
                        Penalty_Sup = 1.0 if ratio_comp <= 1.0 else (comp_all + EPS) / (comp_in_sup + EPS)
                        Sup_Sli_only = sil_sup
                        Sup_Sli_Award = Sup_Sli_only * Award_Sup
                        Sup_Sli_AwardPenalty = Sup_Sli_Award * Penalty_Sup
                else:
                    comp_in_sup = np.nan
                    compl_in_sup = np.nan
                    n_edges_sup = 0
                    Sup_Sli_only = np.nan
                    Sup_Sli_Award = np.nan
                    Sup_Sli_AwardPenalty = np.nan

                record = {
                    "n_components": n_components,
                    "K": k,
                    "bootstrap_id": b + 1,
                    # Red
                    "Red_silhouette": sil_red,
                    "Red_calinski_harabasz": ch_red,
                    "Red_davies_bouldin": db_red,
                    "Red_n_singleton_clusters": n_singleton_red,
                    "Red_n_non_singleton_clusters": n_non_singleton_red,
                    "Red_min_cluster_size_non_singleton": min_size_non_singleton_red,
                    "Red_max_cluster_size": max_size_red,
                    "Red_valid_structure": valid_structure_red,
                    "Red_comp_in": comp_in_red,
                    "Red_compl_in": compl_in_red,
                    "Red_n_edges_within": n_edges_red,
                    "Red_Sli_only": Red_Sli_only,
                    "Red_Sli_Award": Red_Sli_Award,
                    "Red_Sli_AwardPenalty": Red_Sli_AwardPenalty,
                    # Sup
                    "Sup_silhouette": sil_sup,
                    "Sup_calinski_harabasz": ch_sup,
                    "Sup_davies_bouldin": db_sup,
                    "Sup_n_singleton_clusters": n_singleton_sup,
                    "Sup_n_non_singleton_clusters": n_non_singleton_sup,
                    "Sup_min_cluster_size_non_singleton": min_size_non_singleton_sup,
                    "Sup_max_cluster_size": max_size_sup,
                    "Sup_valid_structure": valid_structure_sup,
                    "Sup_comp_in": comp_in_sup,
                    "Sup_compl_in": compl_in_sup,
                    "Sup_n_edges_within": n_edges_sup,
                    "Sup_Sli_only": Sup_Sli_only,
                    "Sup_Sli_Award": Sup_Sli_Award,
                    "Sup_Sli_AwardPenalty": Sup_Sli_AwardPenalty,
                }
                records.append(record)

    df_boot = pd.DataFrame(records)
    boot_csv = os.path.join(OUT_DIR, "ICA_RedSup_bootstrap_raw.csv")
    df_boot.to_csv(boot_csv, index=False)
    print(f"[IO] 已保存 bootstrap 明细到: {boot_csv}")

    summary_rows = []
    group_cols = ["n_components", "K"]
    for (n_components, k), grp in df_boot.groupby(group_cols):
        row = {
            "n_components": n_components,
            "K": k,
            "n_bootstrap_total": len(grp),
        }

        # Red summary
        grp_red_valid = grp[grp["Red_valid_structure"] & grp["Red_Sli_only"].notna()]
        row["Red_n_valid_bootstrap"] = len(grp_red_valid)
        if len(grp_red_valid) > 0:
            row["Red_Sli_only_mean"] = grp_red_valid["Red_Sli_only"].mean()
            row["Red_Sli_only_std"] = grp_red_valid["Red_Sli_only"].std(ddof=1)
            row["Red_Sli_Award_mean"] = grp_red_valid["Red_Sli_Award"].mean()
            row["Red_Sli_AwardPenalty_mean"] = grp_red_valid["Red_Sli_AwardPenalty"].mean()
            row["Red_comp_in_mean"] = grp_red_valid["Red_comp_in"].mean()
            row["Red_compl_in_mean"] = grp_red_valid["Red_compl_in"].mean()
        else:
            row["Red_Sli_only_mean"] = np.nan
            row["Red_Sli_only_std"] = np.nan
            row["Red_Sli_Award_mean"] = np.nan
            row["Red_Sli_AwardPenalty_mean"] = np.nan
            row["Red_comp_in_mean"] = np.nan
            row["Red_compl_in_mean"] = np.nan

        # Sup summary
        grp_sup_valid = grp[grp["Sup_valid_structure"] & grp["Sup_Sli_only"].notna()]
        row["Sup_n_valid_bootstrap"] = len(grp_sup_valid)
        if len(grp_sup_valid) > 0:
            row["Sup_Sli_only_mean"] = grp_sup_valid["Sup_Sli_only"].mean()
            row["Sup_Sli_only_std"] = grp_sup_valid["Sup_Sli_only"].std(ddof=1)
            row["Sup_Sli_Award_mean"] = grp_sup_valid["Sup_Sli_Award"].mean()
            row["Sup_Sli_AwardPenalty_mean"] = grp_sup_valid["Sup_Sli_AwardPenalty"].mean()
            row["Sup_comp_in_mean"] = grp_sup_valid["Sup_comp_in"].mean()
            row["Sup_compl_in_mean"] = grp_sup_valid["Sup_compl_in"].mean()
        else:
            row["Sup_Sli_only_mean"] = np.nan
            row["Sup_Sli_only_std"] = np.nan
            row["Sup_Sli_Award_mean"] = np.nan
            row["Sup_Sli_AwardPenalty_mean"] = np.nan
            row["Sup_comp_in_mean"] = np.nan
            row["Sup_compl_in_mean"] = np.nan

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUT_DIR, "ICA_RedSup_Hyper_Summary.csv")
    df_summary.to_csv(summary_csv, index=False)


if __name__ == "__main__":
    main()
