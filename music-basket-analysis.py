# Market Basket Analysis on Music Streaming Data

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from mlxtend.preprocessing import TransactionEncoder 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
from collections import Counter

df = pd.read_csv("spotify_dataset.csv")
print(df.head()) 

# Display basic information about the dataset
print(df.info())

# Display summary statistics of the dataset
print(df.describe())
print(df.shape)

print(df['Unnamed: 0']) 
df = df.drop(columns=['Unnamed: 0'])
print(df.head()) 

# Check for missing values
print(df.isnull().sum())
df = df.dropna()
print(df.shape) 
df.dropna(subset=['artists', 'album_name', 'track_name'], inplace=True) 
print(df.isna().sum()) 

# Data Transformation
df = df[['track_id', 'album_name', 'artists']]
transactions = df.groupby('artists')['album_name'].apply(list).values.tolist() 
print(transactions[0:5]) 

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns = te.columns_)
print(df_encoded.head()) 
print(df_encoded.shape)
print(df_encoded.sum(axis=1).mean()) 

# Model Building
# The Apriori algorithm identifies item combinations that occur frequently, 
# based on a minimum support threshold 
# the proportion of playlists containing a given combination.
frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True, low_memory=True)
frequent_itemsets.sort_values(by='support', ascending=False)

# The association rules express relationships like:
# “If a playlist contains Song A, it likely also contains Song B.”
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.1)
rules.sort_values(by='lift', ascending=False, inplace=True)
print(rules.head()) 

# visualizing the associations
frequent_itemsets.nlargest(10, 'support').plot(kind='barh', x='itemsets', y='support', legend=False, color='orchid')
plt.title('Top Frequent Song Combinations')
plt.xlabel('Support')
plt.ylabel('Song Sets')
plt.tight_layout()
plt.show()

# Network Graph of Associations
print(rules.shape) 
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
G = nx.from_pandas_edgelist(rules.head(10), 'antecedents', 'consequents', ['lift'])
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, k=0.5, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=2, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
plt.title("Network of Song/Artist Associations", fontsize=14, fontweight='bold')
plt.axis('off')
plt.show() 

# Word Cloud of Most Common Artists
artist_freq = Counter(df['artists'])
wordcloud = WordCloud(width=1000, height=600, background_color='black', colormap='plasma').generate_from_frequencies(artist_freq)
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Artists in the Dataset', fontsize=14)
plt.tight_layout()
plt.show() 

# Save the rules to a CSV file
rules.to_csv("music_association_rules.csv", index=False) 