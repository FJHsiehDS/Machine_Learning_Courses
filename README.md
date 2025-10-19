# Machine Learning Courses

This repository contains a series of notebooks exploring core machine learning algorithms, from clustering and dimensionality reduction, language model and graph analytics. Below is an outline of each notebook and the major concepts demonstrated.

---
## Unsupervised ML:

### **HW2A - Clustering: K-Means and EM for Mixture Models**
- Implemented Expectation–Maximization (EM) for :
  - **Hard & soft K-means** algorithms  
  - **Gaussian Mixture Models (GMM)**  
  - **Binomial mixture model** for discrete data  
- Evaluated clusters using:
  - **Purity** and **Gini index**  
  - **Log-likelihood** progression as a convergence metric  
- **Concepts:**
  - EM algorithm derivation  
  - Soft vs. hard assignments  
  - Mixture model parameter updates (means, covariances, weights)

---

### **HW2B – Density- and Hierarchical-based Clustering**
- Implemented:
  - **DBSCAN** (density-based clustering)  
  - **Agglomerative hierarchical clustering** with linkage matrices  
- Evaluated using:
  - **Silhouette score** for cluster separation  
  - Dendrogram visualization and neighbor-distance plots  
- **Concepts:**
  - Core vs. border vs. noise points in DBSCAN  
  - Cluster distance metrics (single, complete, average linkage)

---

### **HW3A – Kernel PCA & Nonlinear Dimensionality Reduction**
- Implemented **Kernel Principal Component Analysis (Kernel PCA)**.  
- Applied to:
  - **MNIST digit dataset** and **20 Newsgroups text data**  
- Compared linear PCA vs. kernel PCA using RBF kernels.  
- **Concepts:**
  - Kernel trick and feature space mapping  
  - Eigen decomposition in feature space  
  - Dimensionality reduction for nonlinearly separable data

---

### **HW3B – t-SNE Visualization and Perplexity Tuning**
- Implemented **t-SNE** from scratch and with scikit-learn.  
- Visualized embeddings for **20 Newsgroups** and **MNIST** data.  
- Explored **perplexity** parameter and local/global structure tradeoffs.  
- **Concepts:**
  - High-dimensional similarity (Pij) and low-dimensional mapping (Qij)  
  - KL-divergence optimization in t-SNE  
  - Gradient descent and learning rate effects

---

### **HW4 – Neural Embedding and PyTorch Modeling**
- Built a **PyTorch-based word embedding model** (similar to Skip-gram).  
- Trained embeddings on custom text corpus.  
- Visualized latent space using **UMAP**.  
- **Concepts:**
  - Word2Vec embedding learning  
  - Negative sampling and context windows  
  - Neural representation learning and cosine similarity

---

### **HW5 – Topic Modeling and Summarization**
- Implemented **Latent Dirichlet Allocation (LDA)** with Gensim.  
- Performed **document–topic analysis** on DUC and other text corpora.  
- Used **ROUGE metrics** to evaluate text summarization outputs.  
- **Concepts:**
  - Topic–word and document–topic distributions  
  - Perplexity and coherence scores  
  - Automatic summarization and evaluation

---

### **HW5B – Gibbs Sampling and Bayesian Topic Models**
- Implemented **Gibbs sampling** for LDA and Gaussian mixture models.  
- Simulated posterior sampling and burn-in effects.  
- **Concepts:**
  - Sampling from conditional distributions  
  - Hyperparameters α, β in LDA  
  - Convergence and sample variance estimation

---

### **HW6 – Graph Analysis and Community Detection**
- Implemented **graph-based learning and community detection**.  
- Analyzed modularity and connected components.  
- Visualized networks and partitions using **igraph**.  
- **Concepts:**
  - Modularity maximization  
  - Graph partitioning and edge-based scoring  
  - Community structure metrics and visualization

---

## Key Concepts Summary

| Category | Core Concepts |
|-----------|----------------|
| **Clustering** | K-means, EM, GMM, Purity, Gini Index, Silhouette Score |
| **Dimensionality Reduction** | PCA, Kernel PCA, t-SNE, UMAP |
| **Density Methods** | DBSCAN, hierarchical clustering, linkage metrics |
| **Probabilistic Models** | Mixture models, LDA, Gibbs sampling |
| **Neural Models** | Word2Vec, PyTorch embeddings, negative sampling |
| **Graph Models** | Modularity, community detection, network layout |

---
