# Spotify Market Basket Analysis

## Overview  
This project analyzes Spotify listener behavior using Market Basket Analysis (MBA) to uncover co-listening patterns — songs and artists that frequently appear together in playlists.  
The goal is to reveal hidden associations in music preferences and provide insights for smarter playlist generation or recommendation systems.

---

## Workflow  

### 1. Data Cleaning  
- Removed duplicates and missing values.  
- Focused on key columns: `track_id`, `album_name`, and `artists`.  

### 2. Data Transformation  
- Converted playlist data into a transactional format using `TransactionEncoder`.  
- Prepared the dataset for frequent itemset mining.  

### 3. Model Building  
- Applied the Apriori Algorithm (`min_support = 0.001`) to find frequent itemsets.  
- Generated Association Rules using metrics like lift and confidence.  

### 4. Visualization  
- Matplotlib and NetworkX: Song and artist association graphs.  
- Tableau Dashboards:  
  - Top artists by frequency  
  - Common album combinations  
  - Word cloud of popular genres or tracks  

---

## Insights  
- Identified artists often listened to together.  
- Found strong song–album and artist–artist associations.  
- Highlighted clusters of related music useful for recommendation systems.  

---

## Tools and Technologies  
| Category | Tools |
|-----------|--------|
| Data Cleaning & Analysis | Python (pandas, numpy) |
| Association Mining | mlxtend (Apriori, Association Rules) |
| Visualization | Matplotlib, NetworkX, Tableau |
| Dataset | Spotify Tracks Dataset |

---

## Outcome  
Demonstrated how data analytics can uncover patterns in Spotify listening behavior and co-listening trends, supporting more personalized and data-driven music recommendation insights.  

---

## Author  
**Manaswi Priya Maddu**  

---

