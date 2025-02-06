# Amazon Metadata - Co-Purchasing Data Analysis and Recommendation- Community Detection & Link Prediction using Graph Neural Networks
## Introduction
The project analyzes co-purchasing patterns in the Amazon Product Graph Dataset to understand customer behavior and improve marketing strategies. Key methodologies include:  
- Graph construction and visualization.  
- Community detection (Girvan-Newman and Louvain algorithms).  
- Recommendation systems using Jaccard similarity and centrality metrics.  
- Link prediction with GNNs, Logistic Regression, Decision Trees, and Random Forests.  

---

## Dataset Characterization
- **Source**: Stanford Network Analysis Project ([SNAP](https://snap.stanford.edu/data/amazon0302.html)).  
- **Metadata**: 548,552 unique items (books, CDs, DVDs) with sales rank, reviews, and co-purchasing data.  
- **Edges**: 1,788,725 co-purchasing relationships.  
- **Product Distribution**: Books dominate the dataset, making them the focus for recommendations.  

---

## Network Analysis
### Key Metrics
- **Degree Centrality**: Identifies frequently co-purchased products. Top node: `8`.  
- **Betweenness Centrality**: Measures bridge nodes. Formula:  

  $$C_n(v) = \\sum \\frac{\\sigma_s(v)}{\\sigma_a}$$  

  - Where **\\(\\sigma_s(v)\\)** is the number of shortest paths passing through node **v**.  
  - **\\(\\sigma_a\\)** is the total number of shortest paths between nodes. 
- **Clustering Coefficient**: Indicates genre-based communities.  

### Visualization
- Subset: 8,000 entries (2,737 nodes, 6,190 edges).  
- Tools: NetworkX for graph construction, PyVis for visualization.  

---

## Co-Purchasing Analysis
### Community Detection
| Algorithm       | Modularity Score | Top Customers (Community 1)          |  
|-----------------|------------------|---------------------------------------|  
| **Girvan-Newman** | 0.2098          | ID: `3UN6MX5RR02AG` (55 purchases)    |  
| **Louvain**       | **0.79**        | ID: `3UN6MX5RR02AG` (7 purchases)     |  

**Insight**: Louvain outperforms Girvan-Newman in identifying distinct communities.  

---

## Recommendation System
- **Graph Structure**: Nodes = books, edges = co-purchases, edge weights = Jaccard similarity.  
- **Jaccard Similarity**:  
  **
  J(A, B) = \frac{|A \cap B|}{|A \cup B|}
  **
- **Recommendation Workflow**:  
  1. Calculate neighbors of purchased book.  
  2. Sort by average rating and total reviews.  
  3. Select top 5 recommendations.  

**Example Input**:  
- Book: *Bloomsbility* (ASIN: `006440823X`, Avg. Rating: 4.5).  
- **Top Recommendations**: Books with high category similarity and ratings.  

---

## Link Prediction
### Models and Performance
| Model             | Accuracy | MSE   | ROC AUC Score |  
|-------------------|----------|-------|---------------|  
| Logistic Regression | 0.919   | 0.08  | 0.907         |  
| Decision Tree      | 0.954   | 0.045 | 0.945         |  
| **Random Forest**  | **0.964** | **0.035** | **0.948**     |  
| SVM                | 0.94    | 0.0598 | 0.907         |  

**GNN Performance**: Low accuracy (0.0031) due to data sparsity.  

---

## Evaluation Metrics
- **Accuracy**: \( \frac{TP + TN}{TP + TN + FP + FN} \)  
- **MSE**: \( \frac{1}{n} \sum_{i=1}^{n} (y_i - y_i')^2 \)  
- **ROC AUC**: Area under the TPR-FPR curve.  

---

## Results
- **Community Detection**: Louvain achieved higher modularity (0.79 vs. 0.2098).  
- **Recommendations**: Generated using Jaccard similarity and centrality.  
- **Link Prediction**: Traditional ML models outperformed GNNs.  

---

## Dependencies

- Python 3.x
- NetworkX
- Pandas
- Numpy
- torch_geometric
- Scikit-learn
- Matplotlib (for visualization)
  
---

## Contributors
- [Reshma Ananda Prabhakar](https://github.com/reshmaananda/)
- [	Hafeeza Begum Dudekula](https://github.com/HafeezaBegum)
- [Lokesh Poluru Velayudham](https://github.com/lokeshvelayudham/)


