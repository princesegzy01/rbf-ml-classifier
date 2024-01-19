from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

corpus = [
             "Vodafone Wins ₹ 20,000 Crore Tax Arbitration Case Against Government",
             "Voda Idea shares jump nearly 15% as Vodafone wins retro tax case in Hague",
             "Gold prices today fall for 4th time in 5 days, down ₹6500 from last month high",
             "Silver futures slip 0.36% to Rs 59,415 per kg, down over 12% this week",
             "Amazon unveils drone that films inside your home. What could go wrong?",
             "IPHONE 12 MINI PERFORMANCE MAY DISAPPOINT DUE TO THE APPLE B14 CHIP",
             "Delhi Capitals vs Chennai Super Kings: Prithvi Shaw shines as DC beat CSK to post second consecutive win in IPL",
             "French Open 2020: Rafael Nadal handed tough draw in bid for record-equaling 20th Grand Slam"
]

corpus_embeddings = embedder.encode(corpus)

from sklearn.cluster import KMeans

num_clusters = 2
# Define kmeans model
clustering_model = KMeans(n_clusters=num_clusters)

# Fit the embedding with kmeans clustering.
clustering_model.fit(corpus_embeddings)

# Get the cluster id assigned to each news headline.
cluster_assignment = clustering_model.labels_

print(cluster_assignment)
print(corpus_embeddings)