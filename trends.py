from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

class TrendDetector:
    def __init__(self):
        # Simpler configuration for small datasets
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        umap_model = UMAP(n_neighbors=2, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=2, metric='euclidean', prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
        
        self.topic_model = BERTopic(
            embedding_model=sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            verbose=True
        )
        self.is_fitted = False
        self.documents = []
    
    def add_documents(self, new_documents):
        """Add new documents to our corpus"""
        self.documents.extend(new_documents)
        return len(self.documents)
    
    def detect_trends(self):
        """Detect trending topics from collected documents"""
        if len(self.documents) < 3:
            return {"error": "Need at least 3 documents for trend detection"}
        
        try:
            # Fit the model
            topics, probs = self.topic_model.fit_transform(self.documents)
            self.is_fitted = True
            
            # Get topic info
            topic_info = self.topic_model.get_topic_info()
            
            # Get trending topics
            trending_topics = []
            for _, row in topic_info.head(5).iterrows():
                if row['Topic'] != -1:  # Skip outlier topic
                    topic_words = self.topic_model.get_topic(row['Topic'])
                    trending_topics.append({
                        "topic_id": int(row['Topic']),
                        "count": int(row['Count']),
                        "keywords": [word for word, score in topic_words[:5]],
                        "sample_docs": [self.documents[i] for i, t in enumerate(topics) if t == row['Topic']][:2]
                    })
            
            return {
                "total_documents": len(self.documents),
                "total_topics": len(topic_info) - 1,
                "trending_topics": trending_topics
            }
            
        except Exception as e:
            return {"error": f"Trend detection failed: {str(e)}"}
