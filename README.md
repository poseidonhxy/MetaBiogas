# **Representation learning uncovers generalizable microbial pattern in the anaerobic digestion microbiome**#

###Project Aim: Anaerobic digestion (AD) is driven by complex microbial communities, while metagenomics approaches have advanced its characterization. However, existing studies are constrained by site-specific conditions and data sparsity, and the analytical frameworks predominantly rely on relative abundance, which are insufficient to capture ecological organization across diverse AD systems. ###

## **Supervised learning: Prediction Pearson correation between two MAG abundance from their interaction features**

###RF_class: Classification predction  

SVR_prediction: Regression prediction  ###


**Representation learning: Clustering for MAGs based on thier abundance and interaction features**  

Simple-Kmeans: Simple k-means model as the baseline for evaluation  

ICA-Feature Aware Model: Embed abundance and interaction features using ICA  

Encoder-Feature Aware Model: Embed abundance and interaction features using Encoder  

ICA-GNN: Embeding abundance by ICA. Embeded abundance is the node for GNN while interaction is the edge  

Encoder-GNN: Embeding abundance by Encoder. Embeded abundance is the node for GNN while interaction is the edge  

Red_Encoder-GNN: Encoder-GNN framework for extract abundance similar and function similar ecology nich from core MAGs  

