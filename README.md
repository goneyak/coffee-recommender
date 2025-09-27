# ☕ Coffee Recommendation & Visualization

This project builds a **coffee similarity and recommendation system** using the Kaggle dataset:  
[Coffee Reviews (1997–2025)](https://www.kaggle.com/datasets/xinowo/coffee-reviews-feb-1997-mar-2025?select=coffee_reviews_parsed.csv)

---

## 🔎 Project Goal
- Extract a "Flavor DNA" from cupping scores (aroma, acidity, body, flavor, aftertaste).  
- Compare **Flavor-only** vs **Enriched** similarity (metadata: rating, roast, Agtron, origin).  
- Build a simple **recommendation system**.  
- Visualize results with **radar charts, PCA/t-SNE maps, and network analysis**.

---

## 📂 Workflow
1. **Load & clean data**  
   - Handle missing values.  
   - Combine `Acidity` + `Acidity/Structure` → `Acidity_final`.  
   - Compute `Agtron_final` as average of whole & ground.

2. **Similarity computation**  
   - **Flavor-only**: cosine similarity on 5D cupping scores.  
   - **Enriched**: add rating, roast level, Agtron, and continent-level origin.  

   \[
   \vec{g}_i = \lambda_f \cdot z(\vec{f}_i) \;\oplus\; \lambda_m \cdot z(\vec{m}_i) \;\oplus\; \lambda_r \cdot z(\vec{r}_i)
   \]

3. **Recommendation system**  
   - Input: a coffee name.  
   - Output: Top-3 similar coffees (Flavor-only vs Enriched).

4. **Visualization**  
   - Radar charts → Flavor profile comparison.  
   - PCA / t-SNE maps → clustering in taste space.  
   - Network graph → similarity network by continent.

---

## 📊 Example Outputs

### Recommendation
**Input**: Brazil Santos  
- Flavor-only → Colombia Supremo  
- Enriched → Another Brazil Santos (different roaster)

### Radar Chart
Visual comparison of flavor vectors (aroma, acidity, body, flavor, aftertaste).

### PCA Map
- Left: Flavor-only → mixed scatter  
- Right: Enriched → clearer clusters by continent  

### Network Graph
- Nodes = coffees  
- Edges = similarity > 0.9  
- Colors = continent of origin  

---

## ⚙️ Installation
```bash
pip install -r requirements.txt
```

## 🛠️ Main Dependencies
pandas, numpy
scikit-learn
matplotlib, seaborn
networkx

## 📌 Future Work
Incorporate price into similarity weighting
Interactive dashboard (Streamlit)
Deploy as API for coffee recommendations
