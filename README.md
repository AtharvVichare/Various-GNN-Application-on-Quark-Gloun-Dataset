Dowload the 

Raw file from ML4Sci(quark-gloun-classification dataset): https://drive.google.com/file/d/1WO2K-SfU2dntGU4Bb3IYBp9Rh7rtTYEr/view?usp=sharing

And It' processed version (processed with CT-02 (Data Pre Pre Processor in Common Task 2),  the same processed dataset is used for further sp1 and sp4 tasks.

Download processed Dataset: https://drive.google.com/file/d/1ogTlGvCi6QH5E8WfUiH_wEhg3U6mbcdH/view?usp=sharing

The proccessed version contains list of single pyg Data wrapped graphs. Hence repeated raw data to graph conversion is required.

Put these both files in Data folder to run all the code successfully.

# Jets as Graphs: GNN Benchmark

The data preprocessing builds the graph dataset once; the model trainer loads it and benchmarks six GNN architectures and at last the file implements stable chebnet with contrastive learning.

---

###  Data Preprocessor

**Notebook:** `Data_Preprocessing_.ipynb`

**Task:** Convert raw quark/gluon jet images (HDF5, shape 125×125×3) into a PyTorch Geometric graph dataset ready for GNN training.

**Pipeline:**

```
HDF5 input  (X_jets: N×125×125×3,  y,  pt,  m0)
      │
      ▼
Stage 1 — Load & normalise
  Per-channel min-max normalisation across all loaded events

      │
      ▼
Stage 2 — Image → point cloud
  Non-zero pixels → 3D points
  Node features (5-D): [x_norm, y_norm, z, intensity, channel_id]
  z separates calorimeter layers via LAYER_SEP = 0.5
  Top MAX_NODES pixels by intensity retained per event

      │
      ▼
Stage 3 — kNN graph (pure PyTorch, no sklearn)
  torch.cdist → k=8 nearest neighbours
  Undirected edges, no self-loops
  edge_attr = Euclidean distance ΔR

      │
      ▼
Stage 4 — PyG Data object
  x: (N_nodes, 5)   edge_index: (2, E)   edge_attr: (E, 1)
  y, pt, m0 stored per graph

      │
      ▼
InMemoryDataset.collate → saved as jet_pyg_dataset.pt
```

**Config:**

| Parameter | Value |
|---|---|
| N events | 10,000 |
| Max nodes per graph | 1,000 |
| k-NN neighbours | 8 |
| Layer separation (z) | 0.5 |
| Node feature dim | 5 |
| Edge feature dim | 1 (ΔR) |

**Extras:** Interactive 3D Plotly visualiser for per-event graph inspection (node colour = intensity, edges = kNN connectivity).
<img width="1209" height="1023" alt="image" src="https://github.com/user-attachments/assets/1917591d-199b-4d7e-a894-4c1d69260596" />

---

###  GNN Model Training & Benchmark

**Notebook:** `Model_Training.ipynb`

**Task:** Train and benchmark five GNN classifiers on the pre-built graph dataset. Compare against ChebNet (pre-trained, results injected). All models follow the same backbone pattern: 3 graph-conv layers + BatchNorm → global_mean_pool → MLP head (hidden=128).

**Models benchmarked:**

| Model | Conv layers | Notes |
|---|---|---|
| GCN | GCNConv × 3 | Baseline spectral |
| GAT | GATConv × 3 (4 heads) | Attention, dropout=0.2 |
| GraphSAGE | SAGEConv × 3 | Inductive, neighbourhood sampling |
| EdgeConv | EdgeConv × 3 (max aggr) | DGCNN-style, edge MLP |
| ImprovedGNN | GAT→SAGE→GIN + multi-scale pool | add + mean + max pooling |
| **ChebNet** | ChebConv × 3, K=5 | Spectral, pre-trained result |

**Training config:**

| Parameter | Value |
|---|---|
| Split | 70% train / 15% val / 15% test |
| Batch size | 64 |
| Epochs | 50 |
| Early stopping | Patience 10 (val loss) |
| Optimiser | AdamW |
| Scheduler | ReduceLROnPlateau |
| Grad clip | 1.0 |
| Hidden dim | 128 |

**Results (test set):**

| Model | Accuracy | ROC-AUC |
|---|---|---|
| GCN | **69.33%** | **0.7776** |
| GAT | **68.53%** | **0.7753** |
| GraphSAGE | **69.73%** | **0.7740** |
| EdgeConv | **70.80%** | **0.7845** |
| ImprovedGNN | **69.13%** | **0.7692** |
| **ChebNet** | **72.90%** | **0.7869** |

> Trained model results populate at runtime. ChebNet results are injected from pre-training.

**Visualisations produced:**
- Training curves (val accuracy + val AUC per epoch for all trained models)
- ROC curves for all six models on the test set
- Accuracy & AUC summary bar chart
- Confusion matrices for all six models

---
Classifying quark vs gluon jets using a Stable-ChebNet encoder with contrastive learning.
---
### Describing Stable Chebnet
---
Stable ChebNet: ODE Formulation with Antisymmetric Weights
To enable large K without instability, Stable-ChebNet reformulates the layer dynamics as a continuoustime ODE[2]:
<img width="1784" height="400" alt="image" src="https://github.com/user-attachments/assets/5da4072e-8a7a-4943-9aba-451c7aff5736" />

By enforcing antisymmetric weight matrices W⊤k = −Wk and using the symmetric normalised Laplacian, Theorem 3 guarantees purely imaginary Jacobian eigenvalues Re(λi(J)) = 0, ensuring
non-dissipative information propagation: energy is preserved and distant nodes remain sensitive to
far-away inputs.
Discretising with forward Euler and adding a damping term γI yields the Stable-ChebNet update:
<img width="985" height="155" alt="image" src="https://github.com/user-attachments/assets/339d5451-d058-4c6f-9592-6be3fb70406e" />
where ϵ > 0 is the step size. Theorem 4 proves second-order stability:

<img width="445" height="135" alt="image" src="https://github.com/user-attachments/assets/f225e9c7-f106-4dc2-b947-7844da18e746" />

---


<img width="1433" height="627" alt="image" src="https://github.com/user-attachments/assets/2cd0a83c-64ec-4393-8f40-eea75358447e" />


### Stable-ChebNet + Contrastive Learning version 1

**Notebook:** `ST-01.ipynb`

**Architecture:** Stable-ChebNet encoder + GLADC perturbed dual-encoder + NT-Xent contrastive loss + binary classification head.

```
Input jet graph
      │
      ├──────────────────────────────────┐
      ▼                                  ▼
Clean encoder f(θ)              Perturbed encoder f(θ′)
3× StableChebNetLayer(K)        θ′ = θ + σ·ε,  ε~N(0,1)
   X^(l+1) = LayerNorm(          (functional_call API)
     X^(l) + ε·Σ_k T_k(L̃)                | 
     X^(l)(W_k − W_kᵀ − γI))             | 
RELU activation per layer                | 
      │                                  │
  z_graph = cat(max_pool, mean_pool)   ẑ_graph
      │                                  │
      └──────────┬───────────────────────┘
                 │
         Projection head
         Linear → BN → ELU → Linear
                 │
         NT-Xent loss (L_con, τ=0.15)
         + Binary CE loss (L_ce)
         L_total = L_ce + λ(t)·L_con
         λ ramps 0 → final over warmup epochs
                 │
         MLP classifier head (3 layers + BN)
```

**Config:**

| Parameter | Value |
|---|---|
| ChebNet order K | configurable |
| ChebNet layers | 3 |
| Hidden dim | 128 |
| Dual pool | max + mean → 256-dim |
| Projection dim | configurable |
| Perturbation σ | configurable |
| ε (Euler step) | configurable |
| γ (damping) | configurable |
| Batch size | 64 |
| LR schedule | OneCycleLR (peak 3e-3, warm-up → cosine) |
| Contrastive temperature τ | 0.15 |
| Grad clip | 2.0 |
| Activation | RELU |

**Key design decisions:**
- `OneCycleLR` replaces `CosineAnnealingLR` — less aggressive early decay, faster initial convergence
- Batch size 64 over 32 — more in-batch negatives for NT-Xent
- Dual-pool readout (max ∥ mean) — richer global jet descriptor
- `torch.func.functional_call` used for weight-perturbed encoder (autograd-safe)
- Antisymmetric weights: `W_antisym = W − Wᵀ − γI` — guarantees Jacobian stability

**Best test results:**

| Metric | Value |
|---|---|
| Accuracy | **72.90%** |
| ROC-AUC | **0.7869** |
| Precision | 0.7422 |
| Recall | 0.6928 |
| F1 | 0.7166 |

---


## Stable-ChebNet + Contrastive Learning version 2


**Notebook:** `Stable Chebnet on quark gloun with contrastive learning.ipynb`

**Task:** supervised contrastive learning (SupCon), a fixed Chebyshev rescaling, triple-pool readout, and training efficiency improvements.

| Metric | Value |
|---|---|
| Accuracy | **72.47%** |
| ROC-AUC | **0.7999** |
| Precision | 0.7612 |
| Recall | 0.6569 |
| F1 | 0.7052 |




**Evaluation additions:**
- UMAP / t-SNE latent space visualisation before and after training — confirms SupCon creates tighter class clusters
- Linear probe (frozen encoder): Linear SVM + Logistic Regression on frozen embeddings — high linear-probe AUC confirms the encoder has learned linearly separable quark/gluon representations

---

## Overall Results Summary

| Task | Model | Accuracy | ROC-AUC |
|---|---|---|---|
| Stable-ChebNet + Contrastive Learning version 1 | Stable-ChebNet + NT-Xent | 72.90% | 0.7869 |
| Stable-ChebNet + Contrastive Learning version 1 | Stable-ChebNet + Supervised contrastive | 72.47% | 0.7999 |


---

## Dependencies

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-{VERSION}.html
pip install h5py numpy matplotlib scikit-learn plotly tqdm
```

---

## References

[2] Luo et al. — Deep Graph Level Anomaly Detection with Contrastive Learning, *Scientific Reports* 2022  
[3] Hariri et al. — Return of ChebNet, arXiv:2104.01725  
[9] Ghojogh & Ghodsi — Graph Neural Network, ChebNet, GCN, Graph Autoencoder: Tutorial and Survey  
[10] Tang, Li, Yu — ChebNet with Rectified Power Units  
[11] Andrews et al. — End-to-End Jet Classification of Quarks and Gluons with CMS Open Data  
[13] Velickovic et al. — Graph Attention Networks, ICLR 2018  
Khosla et al. — Supervised Contrastive Learning, NeurIPS 2020
