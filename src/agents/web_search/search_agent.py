import asyncio
import httpx
from typing import Dict, List, Optional
import logging
from serpapi import GoogleSearch
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    source: str
    relevance_score: float

class WebSearchAgent:
    def __init__(self, google_api_key: str = None, cse_id: str = None):
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.cse_id = cse_id or os.getenv("GOOGLE_CSE_ID")
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        
    async def search_healthcare_info(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search for healthcare information using Google Search"""
        if not self.google_api_key and not self.serpapi_key:
            logger.warning("No search API keys configured")
            return []
        
        try:
            # Use SerpAPI if available, otherwise fallback to direct search
            if self.serpapi_key:
                return await self._serpapi_search(query, max_results)
            else:
                return await self._google_custom_search(query, max_results)
                
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []
    
    async def _serpapi_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using SerpAPI"""
        try:
            # Add healthcare-specific terms to improve results
            enhanced_query = f"{query} healthcare medical health"
            
            search = GoogleSearch({
                "q": enhanced_query,
                "api_key": self.serpapi_key,
                "num": max_results,
                "safe": "active"
            })
            
            results = search.get_dict()
            search_results = []
            
            if "organic_results" in results:
                for result in results["organic_results"][:max_results]:
                    search_results.append(SearchResult(
                        title=result.get("title", ""),
                        snippet=result.get("snippet", ""),
                        url=result.get("link", ""),
                        source="google_search",
                        relevance_score=self._calculate_healthcare_relevance(
                            result.get("title", "") + " " + result.get("snippet", ""),
                            query
                        )
                    ))
            
            return sorted(search_results, key=lambda x: x.relevance_score, reverse=True)
            
        except Exception as e:
            logger.error(f"SerpAPI search error: {str(e)}")
            return []
    
    async def _google_custom_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback to Google Custom Search API"""
        if not self.google_api_key or not self.cse_id:
            return []
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.cse_id,
                "q": f"{query} healthcare medical",
                "num": max_results,
                "safe": "active"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            
            search_results = []
            
            if "items" in data:
                for item in data["items"]:
                    search_results.append(SearchResult(
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                        url=item.get("link", ""),
                        source="google_custom_search",
                        relevance_score=self._calculate_healthcare_relevance(
                            item.get("title", "") + " " + item.get("snippet", ""),
                            query
                        )
                    ))
            
            return sorted(search_results, key=lambda x: x.relevance_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Google Custom Search error: {str(e)}")
            return []
    
    def _calculate_healthcare_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score for healthcare content"""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Base relevance from query term matches
        query_terms = query_lower.split()
        matching_terms = sum(1 for term in query_terms if term in content_lower)
        base_score = matching_terms / len(query_terms) if query_terms else 0
        
        # Healthcare authority boost
        trusted_sources = [
            "mayo clinic", "webmd", "healthline", "medlineplus", "nih.gov",
            "cdc.gov", "who.int", "medical", "hospital", "health system"
        ]
        
        authority_boost = 0
        for source in trusted_sources:
            if source in content_lower:
                authority_boost += 0.2
        
        # Medical terminology boost
        medical_terms = [
            "treatment", "diagnosis", "symptoms", "medication", "therapy",
            "clinical", "patient", "doctor", "physician", "specialist"
        ]
        
        medical_boost = 0
        for term in medical_terms:
            if term in content_lower:
                medical_boost += 0.1
        
        final_score = min(base_score + authority_boost + medical_boost, 1.0)
        return final_score
    
    async def search_emergency_info(self, query: str) -> List[SearchResult]:
        """Special search for emergency medical information"""
        emergency_query = f"emergency medical {query} immediate care urgent"
        results = await self.search_healthcare_info(emergency_query, max_results=3)
        
        # Boost emergency-related results
        for result in results:
            if any(term in result.title.lower() or term in result.snippet.lower() 
                   for term in ["emergency", "urgent", "immediate", "911"]):
                result.relevance_score *= 1.2
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def filter_trusted_sources(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter results to only include trusted healthcare sources"""
        trusted_domains = [
            "mayoclinic.org", "webmd.com", "healthline.com", "medlineplus.gov",
            "nih.gov", "cdc.gov", "who.int", "medicalnewstoday.com", 
            "health.harvard.edu", "clevelandclinic.org"
        ]
        
        filtered_results = []
        for result in results:
            if any(domain in result.url.lower() for domain in trusted_domains):
                result.relevance_score *= 1.3  # Boost trusted sources
                filtered_results.append(result)
            elif result.relevance_score > 0.7:  # Keep high-relevance results
                filtered_results.append(result)
        
        return sorted(filtered_results, key=lambda x: x.relevance_score, reverse=True)

# Global search agent instance
web_search_agent = WebSearchAgent()
