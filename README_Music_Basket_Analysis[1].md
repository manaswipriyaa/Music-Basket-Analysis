# Music Basket Analysis — Spotify

A data mining project applying Market Basket Analysis to Spotify listening data using the Apriori algorithm to discover co-listened song and artist association rules — visualised in an interactive Tableau dashboard.

---

## Problem Statement

Music streaming platforms recommend songs based on what users listen to together. This project uses association rule mining to uncover patterns in Spotify listening behaviour — which songs or artists are frequently streamed together — and surfaces insights that could power playlist generation and recommendation features.

---

## Dataset

- **Source:** Spotify listening history / Kaggle Spotify dataset
- **Format:** User listening sessions with track and artist metadata
- **Key columns:** User ID, track name, artist name, listen timestamp

---

## Approach

1. **Data Cleaning & Transformation**
   - Removed duplicates, handled nulls, standardised artist/track names
   - Transformed raw listening logs into a transaction format (one row per session)

2. **Exploratory Data Analysis**
   - Top 20 most-listened artists and tracks
   - Listening frequency distributions
   - Session length analysis

3. **Market Basket Analysis (Apriori)**
   - Applied the Apriori algorithm using `mlxtend`
   - Generated frequent itemsets with `min_support = 0.02`
   - Extracted association rules filtered by confidence (≥ 0.3) and lift (≥ 1.2)

4. **Visualisation in Tableau**
   - Interactive dashboard showing top association rules
   - Support vs confidence scatter plot
   - Lift heatmap across top artist pairs

---

## Key Findings

- Artists with the highest co-listen lift scores tend to share genre and tempo
- Strong association rules found between artists in the same genre cluster
- Playlist recommendation potential identified for 50+ artist pairs with lift > 2.0

---

## Metrics Explained

| Metric | What it means |
|---|---|
| **Support** | How often the pair appears together across all sessions |
| **Confidence** | How likely B is listened to when A is listened to |
| **Lift** | How much more likely than random chance the pair co-occurs |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Association Mining | mlxtend (Apriori, association_rules) |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Tableau |
| Notebook | Jupyter Notebook |

---

## Project Structure

```
Music-Basket-Analysis/
│
├── data/
│   └── spotify_data.csv
├── notebooks/
│   └── music_basket_analysis.ipynb
├── tableau/
│   └── music_dashboard.twbx
├── outputs/
│   ├── top_rules.csv
│   ├── support_confidence_plot.png
│   └── lift_heatmap.png
└── README.md
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/manaswipriyaa/Music-Basket-Analysis.git
cd Music-Basket-Analysis

# Install dependencies
pip install pandas numpy mlxtend matplotlib seaborn jupyter

# Launch the notebook
jupyter notebook notebooks/music_basket_analysis.ipynb
```

To view the Tableau dashboard, open `tableau/music_dashboard.twbx` in Tableau Desktop or Tableau Public.

---

## Author

**Manaswi Priya Maddu**
B.Tech — AI & Machine Learning | Acharya Nagarjuna University
[LinkedIn](https://linkedin.com/in/manaswi-priya-2126481b8) | [GitHub](https://github.com/manaswipriyaa)
