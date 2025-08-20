import re
import numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# Initialize embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def build_bm25_index(texts: List[str]) -> BM25Okapi:
    """Build BM25 index for keyword-based retrieval"""
    tokenized_texts = [re.findall(r"\w+", text.lower()) for text in texts]
    return BM25Okapi(tokenized_texts)

async def rank_evidence_advanced(claim: str, evidence_list: List[Dict]) -> List[Dict]:
    """ENHANCED evidence ranking with better weighting"""
    if not evidence_list:
        return []
    
    # Prepare texts for ranking
    texts = []
    for ev in evidence_list:
        text = f"{ev.get('title', '')} {ev.get('snippet', '')}"
        texts.append(text)
        ev['combined_text'] = text
    
    # Build BM25 index
    bm25 = build_bm25_index(texts)
    
    # Get BM25 scores
    tokenized_query = re.findall(r"\w+", claim.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get semantic similarity scores
    claim_embedding = embedding_model.encode([claim])
    text_embeddings = embedding_model.encode(texts)
    similarities = util.cos_sim(claim_embedding, text_embeddings)[0]
    
    # Enhanced scoring with recency and source boost
    for i, evidence in enumerate(evidence_list):
        bm25_score = float(bm25_scores[i]) if i < len(bm25_scores) else 0.0
        semantic_score = float(similarities[i])
        credibility = evidence.get('credibility', 0.5)
        
        # Boost for recent sources and news sites
        source_boost = 1.0
        if evidence.get('source') in ['newsapi', 'google_search']:
            source_boost = 1.2  # Boost news sources
        
        # Boost for high-credibility domains
        url = evidence.get('url', '').lower()
        if any(domain in url for domain in ['timesofindia.com', 'hindustantimes.com', 'indianexpress.com', 'reuters.com', 'bbc.com']):
            source_boost *= 1.3
        
        # Boost for title matches
        title_boost = 1.0
        title_lower = evidence.get('title', '').lower()
        claim_lower = claim.lower()
        if any(word in title_lower for word in claim_lower.split() if len(word) > 3):
            title_boost = 1.5
        
        # Enhanced hybrid scoring
        combined_score = (
            semantic_score * 0.35 +
            bm25_score * 0.25 +
            credibility * 0.25 +
            source_boost * 0.15
        ) * title_boost
        
        evidence['similarity'] = semantic_score
        evidence['bm25_score'] = bm25_score
        evidence['combined_score'] = combined_score
        evidence['source_boost'] = source_boost
        evidence['title_boost'] = title_boost
    
    # Sort by combined score
    ranked = sorted(evidence_list, key=lambda x: x['combined_score'], reverse=True)
    return ranked

def remove_duplicates(evidence_list: List[Dict]) -> List[Dict]:
    """Remove duplicate evidence based on title similarity"""
    seen_titles = set()
    unique = []
    for ev in evidence_list:
        title = ev.get("title", "").lower().strip()
        if not title:
            continue
        
        # More sophisticated deduplication
        title_words = set(re.findall(r'\b\w+\b', title))
        is_duplicate = False
        
        for seen_title in seen_titles:
            seen_words = set(re.findall(r'\b\w+\b', seen_title))
            overlap = len(title_words & seen_words) / max(len(title_words | seen_words), 1)
            if overlap > 0.8:  # 80% word overlap
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_titles.add(title)
            unique.append(ev)
    
    return unique
