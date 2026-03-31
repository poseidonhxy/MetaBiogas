#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ================== CONFIG ==================

# ----------- Input: RA -----------
RA_XLSX = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/251211-Cluster/GNN-Encoder/GNN-Database.xlsx"
RA_SHEET = "RA"
INDEX_COL = "secondary_cluster"

DROP_ALLZERO_SAMPLE_COLS = True
USE_LOG1P = True  # log1p(RA)

# ----------- Input: Pairwise Distance / Competition / Complementarity -----------
DIST_PATH = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/Distance_A abundance/Raw data/Distance_long_with_diag_MAGID.xlsx"
DIST_SHEET = 0

PHYLOMINT_PATH = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/Distance_A abundance/Raw data/PhyloMInt_output.xlsx"
PHYLOMINT_SHEET = "PhyloMInt_output"

# ----------- Input: KEGG feature matrix -----------
KEGG_PATH = "/Users/huangxiaoyan/Desktop/MetaBiogas/Function Analysis/KEGG_pathway/MAG vs Class.xlsx"
KEGG_SHEET = "Pathway"

# ----------- Output -----------
OUT_DIR = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/251211-Cluster/Simple_Kmeans"
OUT_XLSX_NAME = "Final_KMeans_log1p_RA.xlsx"

# ----------- KMeans settings -----------
AUTO_SELECT_K = True
K_MIN = 3
K_MAX = 50

SEED = 42
KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 500

# ----------- Constraint -----------
TARGET_NON_SINGLETON_CLUSTERS = 5  #

# ----------- Embedding -----------
PCA_RANDOM_STATE = 20251218
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 20251218

# ----------- Details -----------
SAVE_DETAILS_BYCLUSTER = True


# ================== Utils ==================

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def read_ra_matrix(path: str, sheet: str, index_col: str):
    df = pd.read_excel(path, sheet_name=sheet)
    if index_col in df.columns:
        df = df.set_index(index_col)
    else:
        df = df.set_index(df.columns[0])

    df.index = df.index.astype(str).str.strip()

    Xraw = df.select_dtypes(include=[np.number]).copy()
    if DROP_ALLZERO_SAMPLE_COLS:
        Xraw = Xraw.loc[:, (Xraw.sum(axis=0) > 0)]
    if Xraw.shape[1] == 0:
        raise ValueError("check RA")

    X = np.log1p(Xraw) if USE_LOG1P else Xraw
    return Xraw, X

def count_non_singleton_clusters(labels: np.ndarray) -> int:
    _, counts = np.unique(labels, return_counts=True)
    return int(np.sum(counts > 1))

def cluster_size_table(labels: np.ndarray):
    uniq, counts = np.unique(labels, return_counts=True)
    return pd.DataFrame({"Cluster": uniq.astype(int), "Size": counts.astype(int)}).sort_values("Cluster")

def evaluate_scores(X: np.ndarray, labels: np.ndarray):
    if len(np.unique(labels)) > 1:
        sil = float(silhouette_score(X, labels))
        ch = float(calinski_harabasz_score(X, labels))
    else:
        sil, ch = -1.0, -1.0
    db = float(davies_bouldin_score(X, labels))
    return sil, ch, db

def to_undirected_pair(df: pd.DataFrame, g1_col: str, g2_col: str):
    g1_list, g2_list = [], []
    for a, b in zip(df[g1_col].astype(str), df[g2_col].astype(str)):
        aa, bb = sorted([a.strip(), b.strip()])
        g1_list.append(aa)
        g2_list.append(bb)
    out = df.copy()
    out["g1"] = g1_list
    out["g2"] = g2_list
    return out

def load_pairwise_metrics():
    # Distance
    df_dist = pd.read_excel(DIST_PATH, sheet_name=DIST_SHEET)
    cols = {str(c).strip().lower(): c for c in df_dist.columns}
    if "genome1" not in cols or "genome2" not in cols:
        raise ValueError("Distance canot find Genome1/Genome2 ")
    dist_col = None
    for key in ["distance", "dist"]:
        if key in cols:
            dist_col = cols[key]
            break
    if dist_col is None:
        raise ValueError("Distance canot find Distance ")

    df_dist = df_dist.rename(columns={
        cols["genome1"]: "Genome1",
        cols["genome2"]: "Genome2",
        dist_col: "Distance",
    })[["Genome1", "Genome2", "Distance"]]

    # PhyloMInt
    df_phy = pd.read_excel(PHYLOMINT_PATH, sheet_name=PHYLOMINT_SHEET)
    needed = {"Genome1", "Genome2", "Competition", "Complementarity"}
    if not needed.issubset(df_phy.columns):
        raise ValueError(f"PhyloMInt mising: {needed - set(df_phy.columns)}")
    df_phy = df_phy[["Genome1", "Genome2", "Competition", "Complementarity"]].copy()

    # undirected + aggregate mean
    df_dist = to_undirected_pair(df_dist, "Genome1", "Genome2")
    df_phy = to_undirected_pair(df_phy, "Genome1", "Genome2")

    df_dist_agg = df_dist.groupby(["g1", "g2"], as_index=False)["Distance"].mean()
    df_phy_agg = df_phy.groupby(["g1", "g2"], as_index=False)[["Competition", "Complementarity"]].mean()

    df_pairs = pd.merge(df_dist_agg, df_phy_agg, on=["g1", "g2"], how="outer")
    return df_pairs

def compute_pairwise_stats_by_cluster(genomes, labels, df_pairs):
    cl_df = pd.DataFrame({"Genome": genomes, "Cluster": labels.astype(int)})
    gset = set(cl_df["Genome"].astype(str))

    df_pairs_sub = df_pairs[
        df_pairs["g1"].isin(gset) & df_pairs["g2"].isin(gset) & (df_pairs["g1"] != df_pairs["g2"])
    ].copy()


    cl_map = dict(zip(cl_df["Genome"].astype(str), cl_df["Cluster"]))
    df_pairs_sub["cluster1"] = df_pairs_sub["g1"].map(cl_map)
    df_pairs_sub["cluster2"] = df_pairs_sub["g2"].map(cl_map)
    df_pairs_sub = df_pairs_sub.dropna(subset=["cluster1", "cluster2"])

    df_within = df_pairs_sub[df_pairs_sub["cluster1"] == df_pairs_sub["cluster2"]].copy()

    records = []
    for c_id, sub in df_within.groupby("cluster1"):
        records.append({
            "Cluster": int(c_id),
            "n_pairs": int(sub.shape[0]),
            "mean_Competition": float(sub["Competition"].mean()) if "Competition" in sub.columns else np.nan,
            "mean_Complementarity": float(sub["Complementarity"].mean()) if "Complementarity" in sub.columns else np.nan,
            "mean_Distance": float(sub["Distance"].mean()) if "Distance" in sub.columns else np.nan,
        })

    # ALL baseline
    records.append({
        "Cluster": "ALL",
        "n_pairs": int(df_pairs_sub.shape[0]),
        "mean_Competition": float(df_pairs_sub["Competition"].mean()) if "Competition" in df_pairs_sub.columns else np.nan,
        "mean_Complementarity": float(df_pairs_sub["Complementarity"].mean()) if "Complementarity" in df_pairs_sub.columns else np.nan,
        "mean_Distance": float(df_pairs_sub["Distance"].mean()) if "Distance" in df_pairs_sub.columns else np.nan,
    })

    return pd.DataFrame(records)

def summarize_pairwise_weighted_means(df_bycluster: pd.DataFrame):
    base = df_bycluster[df_bycluster["Cluster"] == "ALL"]
    if base.empty:
        raise ValueError("pairwise by-cluster 表缺少 ALL baseline。")

    base_pairs_all = float(base["n_pairs"].iloc[0])
    base_comp = float(base["mean_Competition"].iloc[0])
    base_compl = float(base["mean_Complementarity"].iloc[0])
    base_dist = float(base["mean_Distance"].iloc[0])

    sub = df_bycluster[df_bycluster["Cluster"] != "ALL"].copy()
    total_pairs_within = float(sub["n_pairs"].sum())

    sum_pairs_times_comp = float(np.nansum(sub["n_pairs"] * sub["mean_Competition"]))
    sum_pairs_times_compl = float(np.nansum(sub["n_pairs"] * sub["mean_Complementarity"]))
    sum_pairs_times_dist = float(np.nansum(sub["n_pairs"] * sub["mean_Distance"]))

    w_comp = sum_pairs_times_comp / total_pairs_within
    w_compl = sum_pairs_times_compl / total_pairs_within
    w_dist = sum_pairs_times_dist / total_pairs_within

    return pd.DataFrame([{
        "MetricType": "Pairwise",
        "total_pairs_within_clusters": total_pairs_within,
        "weighted_mean_Competition_within": w_comp,
        "weighted_mean_Complementarity_within": w_compl,
        "weighted_mean_Distance_within": w_dist,
        "baseline_total_pairs_all": base_pairs_all,
        "baseline_mean_Competition_all": base_comp,
        "baseline_mean_Complementarity_all": base_compl,
        "baseline_mean_Distance_all": base_dist,
        "delta_weighted_Competition_within_minus_all": w_comp - base_comp,
        "delta_weighted_Complementarity_within_minus_all": w_compl - base_compl,
        "delta_weighted_Distance_within_minus_all": w_dist - base_dist,
    }])

def load_feature_matrix(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    genome_col = df.columns[0]
    df = df.copy()
    df[genome_col] = df[genome_col].astype(str).str.strip()
    df = df.set_index(genome_col)
    num_df = df.select_dtypes(include=[np.number]).copy()
    return num_df

def compute_cosine_similarity_by_cluster(df_feat, genomes, labels, feature_type: str):
    cl_df = pd.DataFrame({"Genome": genomes, "Cluster": labels.astype(int)})
    common = sorted(set(cl_df["Genome"]) & set(df_feat.index.astype(str)))
    feat_all = df_feat.loc[common].copy()

    records = []
    for c_id, sub in cl_df.groupby("Cluster"):
        g_list = [g for g in sub["Genome"].astype(str) if g in feat_all.index]
        n = len(g_list)
        if n < 2:
            continue
        mat = feat_all.loc[g_list].values
        sim = cosine_similarity(mat)
        triu = np.triu_indices(n, k=1)
        vals = sim[triu]
        if vals.size == 0:
            continue
        records.append({
            "FeatureType": feature_type,
            "Cluster": int(c_id),
            "n_genomes": int(n),
            "n_pairs": int(n * (n - 1) / 2),
            "mean_cosine_similarity": float(np.mean(vals)),
        })

    # ALL baseline
    mat_all = feat_all.values
    n_all = mat_all.shape[0]
    sim_all = cosine_similarity(mat_all)
    vals_all = sim_all[np.triu_indices(n_all, k=1)]
    records.append({
        "FeatureType": feature_type,
        "Cluster": "ALL",
        "n_genomes": int(n_all),
        "n_pairs": int(n_all * (n_all - 1) / 2),
        "mean_cosine_similarity": float(np.mean(vals_all)) if vals_all.size > 0 else np.nan,
    })

    return pd.DataFrame(records)

def summarize_feature_weighted_means(df_bycluster: pd.DataFrame):
    base = df_bycluster[df_bycluster["Cluster"] == "ALL"]
    if base.empty:
        raise ValueError("feature by-cluster 表缺少 ALL baseline。")
    base_n_pairs = float(base["n_pairs"].iloc[0])
    base_mean = float(base["mean_cosine_similarity"].iloc[0])

    sub = df_bycluster[df_bycluster["Cluster"] != "ALL"].copy()
    total_pairs_within = float(sub["n_pairs"].sum())
    if total_pairs_within <= 0:
        return pd.DataFrame([{
            "MetricType": "CosineSimilarity",
            "FeatureType": df_bycluster["FeatureType"].iloc[0],
            "total_pairs_within_clusters": 0.0,
            "weighted_mean_cosine_within": np.nan,
            "baseline_n_pairs_all": base_n_pairs,
            "baseline_mean_cosine_all": base_mean,
            "delta_weighted_mean_within_minus_all": np.nan,
        }])

    total_sim_within = float(np.nansum(sub["n_pairs"] * sub["mean_cosine_similarity"]))
    w_mean = total_sim_within / total_pairs_within

    return pd.DataFrame([{
        "MetricType": "CosineSimilarity",
        "FeatureType": df_bycluster["FeatureType"].iloc[0],
        "total_pairs_within_clusters": total_pairs_within,
        "weighted_mean_cosine_within": w_mean,
        "baseline_n_pairs_all": base_n_pairs,
        "baseline_mean_cosine_all": base_mean,
        "delta_weighted_mean_within_minus_all": w_mean - base_mean,
    }])

def compute_pca_tsne(X: np.ndarray):
    pca = PCA(n_components=2, random_state=PCA_RANDOM_STATE)
    Z_pca = pca.fit_transform(X)

    n = X.shape[0]
    perplex = float(min(TSNE_PERPLEXITY, max(2, (n - 1) // 3)))
    tsne = TSNE(
        n_components=2,
        perplexity=perplex,
        n_iter=TSNE_N_ITER,
        random_state=TSNE_RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    )
    Z_tsne = tsne.fit_transform(X)
    return Z_pca, Z_tsne, perplex


# ================== Main ==================

def main():
    ensure_outdir(OUT_DIR)
    out_xlsx = os.path.join(OUT_DIR, OUT_XLSX_NAME)

    # ---- 1) Load RA & preprocess (log1p only) ----
    Xraw, X = read_ra_matrix(RA_XLSX, RA_SHEET, INDEX_COL)
    genomes = X.index.astype(str).tolist()
    X_np = X.values

    best = None  # (sil, k, labels, ch, db, non_singleton)
    if FIXED_K is not None:
        k_list = [int(FIXED_K)]
    else:
        if not AUTO_SELECT_K:
            raise ValueError("请设置 FIXED_K，或把 AUTO_SELECT_K=True。")
        k_list = list(range(int(K_MIN), int(K_MAX) + 1))
    for k in k_list:
        km = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
            init="k-means++",
        )
        labels = km.fit_predict(X_np)

        non_singleton = count_non_singleton_clusters(labels)
        if non_singleton != TARGET_NON_SINGLETON_CLUSTERS:
            continue

        sil, ch, db = evaluate_scores(X_np, labels)
        cand = (sil, k, labels, ch, db, non_singleton)

        if best is None or cand[0] > best[0]:
            best = cand



    sil, K_final, labels_final, ch, db, non_singleton = best

    # ---- 3) Pairwise weighted means: Distance/Competition/Complementarity ----
    df_pairs = load_pairwise_metrics()
    pair_bycluster = compute_pairwise_stats_by_cluster(genomes, labels_final, df_pairs)
    pair_weighted = summarize_pairwise_weighted_means(pair_bycluster)

    # ---- 4) Cosine similarity weighted means: RA + KEGG ----
    # 4.1 RA similarity: 用当前 log1p(RA) 矩阵（X）
    ra_feat = X.copy()
    ra_bycluster = compute_cosine_similarity_by_cluster(ra_feat, genomes, labels_final, "RA_log1p")
    ra_weighted = summarize_feature_weighted_means(ra_bycluster)

    # 4.2 KEGG similarity
    kegg_feat = load_feature_matrix(KEGG_PATH, KEGG_SHEET)
    kegg_bycluster = compute_cosine_similarity_by_cluster(kegg_feat, genomes, labels_final, "KEGG")
    kegg_weighted = summarize_feature_weighted_means(kegg_bycluster)

    weighted_means = pd.concat([pair_weighted, ra_weighted, kegg_weighted], ignore_index=True)

    # ---- 5) Embeddings: PCA2D + TSNE2D (on X=log1p RA) ----
    Z_pca, Z_tsne, perplex_used = compute_pca_tsne(X_np)

    df_pca2d = pd.DataFrame(Z_pca, columns=["PCA1", "PCA2"])
    df_pca2d.insert(0, "Genome", genomes)
    df_pca2d["Cluster"] = labels_final.astype(int)

    df_tsne2d = pd.DataFrame(Z_tsne, columns=["TSNE1", "TSNE2"])
    df_tsne2d.insert(0, "Genome", genomes)
    df_tsne2d["Cluster"] = labels_final.astype(int)

    # ---- 6) Labels / ClusterSizes / Metrics ----
    df_labels = pd.DataFrame({"Genome": genomes, "Cluster": labels_final.astype(int)})
    df_cluster_sizes = cluster_size_table(labels_final)

    df_metrics = pd.DataFrame([{
        "K_final": int(K_final),
        "TARGET_non_singleton_clusters": int(TARGET_NON_SINGLETON_CLUSTERS),
        "non_singleton_clusters_observed": int(non_singleton),
        "silhouette": float(sil),
        "calinski_harabasz": float(ch),
        "davies_bouldin": float(db),
        "N_genomes": int(X_np.shape[0]),
        "N_samples_used": int(X_np.shape[1]),
        "TSNE_perplexity_used": float(perplex_used),
    }])

    df_params = pd.DataFrame([{
        "RA_XLSX": RA_XLSX,
        "RA_SHEET": RA_SHEET,
        "INDEX_COL": INDEX_COL,
        "DROP_ALLZERO_SAMPLE_COLS": DROP_ALLZERO_SAMPLE_COLS,
        "USE_LOG1P": USE_LOG1P,
        "AUTO_SELECT_K": AUTO_SELECT_K,
        "K_MIN": K_MIN,
        "K_MAX": K_MAX,
        "SEED": SEED,
        "KMEANS_N_INIT": KMEANS_N_INIT,
        "KMEANS_MAX_ITER": KMEANS_MAX_ITER,
        "DIST_PATH": DIST_PATH,
        "PHYLOMINT_PATH": PHYLOMINT_PATH,
        "KEGG_PATH": KEGG_PATH,
        "OUT_DIR": OUT_DIR,
    }])

    # ---- 7) Write Excel ----
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_params.to_excel(writer, sheet_name="Params", index=False)
        df_metrics.to_excel(writer, sheet_name="Metrics", index=False)
        weighted_means.to_excel(writer, sheet_name="WeightedMeans", index=False)
        df_pca2d.to_excel(writer, sheet_name="PCA2D", index=False)
        df_tsne2d.to_excel(writer, sheet_name="TSNE2D", index=False)
        df_labels.to_excel(writer, sheet_name="Labels", index=False)
        df_cluster_sizes.to_excel(writer, sheet_name="ClusterSizes", index=False)

        if SAVE_DETAILS_BYCLUSTER:
            pair_bycluster.to_excel(writer, sheet_name="Pairwise_ByCluster", index=False)
            ra_bycluster.to_excel(writer, sheet_name="RA_ByCluster", index=False)
            kegg_bycluster.to_excel(writer, sheet_name="KEGG_ByCluster", index=False)

    print(f"[DONE] Final KMeans written: {out_xlsx}")
    print(f"[DONE] K_final={K_final}, silhouette={sil:.4f}, non_singleton={non_singleton}")


if __name__ == "__main__":
    main()
