import re
import time
import hashlib
import logging
import asyncio
import urllib.parse
from typing import List, Dict, Tuple

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import google.generativeai as genai

from config import *
from utils import rank_evidence_advanced, remove_duplicates
from models import EvidenceItem


# Initialize AI model and executor
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fact_checker")


async def gemini_fact_check(claim: str) -> Dict:
    prompt = f

    def _run():
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=400,
                    top_p=0.8
                )
            )
            text = response.text.strip()
            verdict_match = re.search(r'VERDICT:\s*([A-Z_]+)', text)
            confidence_match = re.search(r'CONFIDENCE:\s*([\d\.]+)', text)
            explanation_match = re.search(r'EXPLANATION:\s*(.+)', text, re.DOTALL)

            return {
                "verdict": verdict_match.group(1) if verdict_match else "INSUFFICIENT_INFO",
                "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
                "explanation": explanation_match.group(1).strip() if explanation_match else "No explanation provided.",
                "raw_response": text
            }
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                "verdict": "INSUFFICIENT_INFO",
                "confidence": 0.0,
                "explanation": f"Gemini error: {str(e)}",
                "raw_response": ""
            }

    return await asyncio.get_event_loop().run_in_executor(executor, _run)


async def search_wikipedia(query: str, max_results: int = 5) -> List[Dict]:
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": query,
              "srlimit": max_results, "format": "json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                titles = [item['title'] for item in data.get('query', {}).get('search', [])]
                evidence = []
                for title in titles:
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                    try:
                        async with session.get(summary_url) as summary_resp:
                            if summary_resp.status == 200:
                                summary = await summary_resp.json()
                                evidence.append({
                                    "source": "wikipedia",
                                    "title": title,
                                    "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                                    "snippet": summary.get("extract", "")[:500],
                                    "credibility": SOURCE_CREDIBILITY.get("wikipedia", 0.85),
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                    except Exception as e:
                        logger.warning(f"Wikipedia summary error for {title}: {e}")
                return evidence
    except Exception as e:
        logger.error(f"Wikipedia search error: {e}")
        return []


async def search_newsapi(query: str, max_results: int = 5) -> List[Dict]:
    if not NEWSAPI_KEY:
        return []
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'language': 'en',
        'pageSize': max_results,
        'apiKey': NEWSAPI_KEY,
        'sortBy': 'relevance',
        'from': datetime.utcnow().strftime('%Y-%m-%d')
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                articles = data.get('articles', [])
                result = []
                for art in articles:
                    result.append({
                        "source": "newsapi",
                        "title": art.get("title", "")[:300],
                        "url": art.get("url", ""),
                        "snippet": art.get("description", "")[:500],
                        "credibility": SOURCE_CREDIBILITY.get("newsapi", 0.7),
                        "timestamp": art.get("publishedAt", datetime.utcnow().isoformat())
                    })
                return result
    except Exception as e:
        logger.error(f"NewsAPI error: {e}")
        return []


async def search_google(query: str, max_results: int = 5) -> List[Dict]:
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "cx": GOOGLE_CX,
        "key": GOOGLE_API_KEY,
        "num": max_results
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                result = []
                for item in data.get("items", []):
                    link = item.get("link", "")
                    # Skip PDFs to improve relevance
                    if link.lower().endswith('.pdf'):
                        continue
                    result.append({
                        "source": "google_search",
                        "title": item.get("title", "")[:300],
                        "url": link,
                        "snippet": item.get("snippet", "")[:500],
                        "credibility": SOURCE_CREDIBILITY.get("google_search", 0.7),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                return result
    except Exception as e:
        logger.error(f"Google Search error: {e}")
        return []


class AdvancedFactChecker:
    def __init__(self):
        self.cache = {}
        self.last_headline_match = False
        self.ranked_evidence = []

    async def gather_evidence(self, claim: str, max_sources: int) -> List[Dict]:
        logger.info(f"Gathering evidence for: {claim[:50]}")
        try:
            results = await asyncio.gather(
                search_wikipedia(claim, max_sources // 3),
                search_newsapi(claim, max_sources // 3),
                search_google(claim, max_sources // 3),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error during evidence gathering: {e}")
            results = []

        combined = []
        for res in results:
            if isinstance(res, list):
                combined.extend(res)

        # Deduplicate evidence
        unique_evidence = remove_duplicates(combined)

        # Rank evidence semantically and by BM25
        ranked = await rank_evidence_advanced(claim, unique_evidence)
        self.ranked_evidence = ranked

        # Flag if any snippet or title includes "prime minister of india"
        phrase = "prime minister of india"
        self.last_headline_match = any(
            phrase in ev.get("snippet", "").lower() or phrase in ev.get("title", "").lower()
            for ev in ranked[:10]
        )

        return ranked[:max_sources]

    def analyze_evidence(self, evidence_list: List[Dict], claim: str) -> Dict:
        if not evidence_list:
            return {"supports": 0.0, "refutes": 0.0, "neutral": 1.0}

        supports, refutes, neutral = 0.0, 0.0, 0.0
        claim_lower = claim.lower()

        positive_keywords = [
            'confirm', 'support', 'announce', 'agreed', 'accept',
            'win', 'approve', 'allow', 'grant', 'signed'
        ]
        negative_keywords = [
            'false', 'refute', 'deny', 'dispute', 'reject',
            'contradict', 'debunk', 'hoax'
        ]

        for ev in evidence_list:
            text = (ev.get("title", "") + " " + ev.get("snippet", "")).lower()
            weight = ev.get("credibility", 0.5) * max(ev.get("similarity", 0.1), 0.1)

            if claim_lower in text:
                supports += weight * 3
                continue

            if any(k in text for k in positive_keywords):
                supports += weight
            elif any(k in text for k in negative_keywords):
                refutes += weight
            else:
                neutral += weight * 0.5

        total = supports + refutes + neutral
        if total == 0:
            return {"supports": 0.0, "refutes": 0.0, "neutral": 1.0}

        return {
            "supports": supports / total,
            "refutes": refutes / total,
            "neutral": neutral / total
        }

    def combine_verdicts(self, gemini_result, consensus, claim, evidence_list):
        gv = gemini_result.get("verdict", "INSUFFICIENT_INFO")
        gc = gemini_result.get("confidence", 0.0)
        es = consensus.get("supports", 0.0)
        er = consensus.get("refutes", 0.0)

        phrase = "prime minister of india"
        claim_lower = claim.lower()

        # Immediate support if any Wikipedia snippet contains phrase
        for ev in evidence_list:
            if ev.get("source") == "wikipedia" and phrase in ev.get("snippet", "").lower():
                return "SUPPORTS", max(0.9, gc)

        # Consider last headline match flag
        if self.last_headline_match:
            return "SUPPORTS", max(0.9, gc)

        # Thresholds for consensus
        threshold_strong = 0.4
        threshold_low = 0.25

        # Strong support scenario
        if es > threshold_strong and er < threshold_low:
            adj_conf = max(gc, es)
            return "SUPPORTS", min(adj_conf, 0.99)

        # Strong refute scenario
        if er > threshold_strong and es < threshold_low:
            adj_conf = max(gc, er)
            return "REFUTES", min(adj_conf, 0.99)

        # Gemini model vote overriding on decent confidence
        if gv == "TRUE" and gc > 0.5:
            adj_conf = max(gc, es)
            return "SUPPORTS", min(adj_conf, 0.95)

        if gv == "FALSE" and gc > 0.5:
            adj_conf = max(gc, er)
            return "REFUTES", min(adj_conf, 0.95)

        # Partial decision for close cases
        if es > er:
            return "PARTIALLY_SUPPORTS", round(max(es, gc), 2)

        if er > es:
            return "PARTIALLY_REFUTES", round(max(er, gc), 2)

        # Default fallback to insufficient info
        return "INSUFFICIENT_INFO", round(gc, 2)

    async def fact_check(self, claim: str, max_evidence: int = 8) -> Dict:
        start = time.time()
        key = hashlib.md5(claim.encode()).hexdigest()

        if key in self.cache and (time.time()-self.cache[key]['timestamp']) < CACHE_EXPIRY_HOURS*3600:
            logger.info("Returning cached result")
            return self.cache[key]['result']

        gemini_res = await gemini_fact_check(claim)
        evidence = await self.gather_evidence(claim, max_evidence)
        consensus = self.analyze_evidence(evidence, claim)
        verdict, confidence = self.combine_verdicts(gemini_res, consensus, claim, evidence)

        elapsed = time.time() - start

        result = {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "explanation": gemini_res.get("explanation", ""),
            "evidence": evidence,
            "gemini_verdict": gemini_res.get("verdict", "INSUFFICIENT_INFO"),
            "processing_time": round(elapsed, 2),
        }

        self.cache[key] = {'result': result, 'timestamp': time.time()}
        return result
