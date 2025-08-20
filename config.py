import os
from typing import Dict

# API Configuration
GEMINI_API_KEY = "AIzaSyCYVp6yfDYGfUTGdbeuibEp5X5-WYfRDmg"
NEWSAPI_KEY = "63f95b520c494716b3cd20bcba008aaf"
GOOGLE_API_KEY = "AIzaSyCpOhGZKqeJto2eFt6AP8CWzgfKG2Ji2bY"
GOOGLE_CX = "639dbb54563c64cf0"

# System Configuration
MAX_WORKERS = min(64, (os.cpu_count() or 4) * 4)
MAX_OUTPUT_TOKENS = 400
DEFAULT_CONCURRENCY_LIMIT = 16
SIMILARITY_THRESHOLD = 0.7
BM25_TOP_K = 15
DEFAULT_PASS1_K = 20
DEFAULT_PASS2_K = 10

# Enhanced source credibility weights
SOURCE_CREDIBILITY: Dict[str, float] = {
    "gemini": 1.0,
    "wikipedia": 0.85,
    "pubmed": 0.95,
    "newsapi": 0.8,  # Increased for current news
    "google_search": 0.75,  # Slightly increased
    "times_of_india": 0.9,  # High for major Indian news
    "bbc": 0.95,
    "reuters": 0.95,
    "hindustantimes": 0.85,
    "indianexpress": 0.85,
}

# Cache settings
CACHE_EXPIRY_HOURS = 1

# Enhanced patterns for better matching
SUPPORTING_PATTERNS = [
    'confirms', 'proves', 'demonstrates', 'shows', 'supports', 
    'announced', 'introduces', 'nominated', 'selected', 'appointed',
    'will manufacture', 'accepted proposal', 'agreed to', 'signed',
    'passed', 'approved', 'authorized', 'launched', 'released'
]

REFUTING_PATTERNS = [
    'false', 'myth', 'debunked', 'incorrect', 'no evidence', 
    'denies', 'rejects', 'disputes', 'contradicts', 'canceled',
    'postponed', 'withdrawn', 'reversed', 'overturned'
]
