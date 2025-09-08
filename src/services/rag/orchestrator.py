import asyncio
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from src.agents.query_analyzer.analyzer import QueryAnalyzer
from src.agents.knowledge_retrieval.retriever import knowledge_retriever
from src.agents.web_search.search_agent import web_search_agent
from src.agents.response_generator.generator import response_generator

logger = logging.getLogger(__name__)

class RAGOrchestrator:
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.knowledge_retriever = knowledge_retriever
        self.web_search_agent = web_search_agent
        self.response_generator = response_generator
        
    async def process_query(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Main orchestration method for processing healthcare queries"""
        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            # Step 1: Analyze query intent and extract entities
            analysis = self.query_analyzer.analyze_query(query)
            logger.info(f"Query analysis: {analysis['intent']}, confidence: {analysis['confidence']}")
            
            # Step 2: Handle emergency queries immediately
            if analysis['is_emergency']:
                logger.warning("Emergency query detected - prioritizing immediate response")
                return await self._handle_emergency_query(query, analysis)
            
            # Step 3: Retrieve knowledge from internal database
            kb_results = await self.knowledge_retriever.retrieve_relevant_data(
                query, analysis['intent'], analysis['entities']
            )
            logger.info(f"Retrieved {len(kb_results)} knowledge base results")
            
            # Step 4: Determine if web search is needed
            web_results = []
            if self._should_search_web(analysis, kb_results):
                logger.info("Performing web search for additional information")
                web_results = await self.web_search_agent.search_healthcare_info(query)
                logger.info(f"Retrieved {len(web_results)} web search results")
            
            # Step 5: Generate comprehensive response
            response = await self.response_generator.generate_response(
                query=query,
                intent=analysis['intent'],
                kb_results=kb_results,
                web_results=web_results,
                confidence=analysis['confidence']
            )
            
            # Step 6: Add orchestration metadata
            response.update({
                'query_analysis': {
                    'intent': analysis['intent'],
                    'entities': analysis['entities'],
                    'confidence': analysis['confidence']
                },
                'data_sources': {
                    'knowledge_base': len(kb_results) > 0,
                    'web_search': len(web_results) > 0,
                    'kb_results_count': len(kb_results),
                    'web_results_count': len(web_results)
                },
                'processing_time': datetime.now().isoformat(),
                'user_id': user_id
            })
            
            logger.info(f"Query processed successfully, final confidence: {response.get('confidence', 0)}")
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG orchestration: {str(e)}")
            return {
                'status': 'error',
                'answer': 'I apologize, but I encountered a technical issue. Please contact our customer service at 0800-123-456 for assistance.',
                'confidence': 0.0,
                'escalated': True,
                'customer_service_info': {
                    'phone': '0800-123-456',
                    'reason': 'system_error'
                },
                'timestamp': datetime.now().isoformat()
            }
    
    async def _handle_emergency_query(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Special handling for emergency medical queries"""
        
        # Get some basic information but prioritize emergency response
        kb_task = asyncio.create_task(
            self.knowledge_retriever.retrieve_relevant_data(
                query, analysis['intent'], analysis['entities']
            )
        )
        
        web_task = asyncio.create_task(
            self.web_search_agent.search_emergency_info(query)
        )
        
        # Wait for both but don't delay too long
        try:
            kb_results, web_results = await asyncio.wait_for(
                asyncio.gather(kb_task, web_task), timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("Emergency query processing timeout - using emergency response")
            kb_results, web_results = [], []
        
        return await self.response_generator.generate_response(
            query=query,
            intent="emergency",
            kb_results=kb_results,
            web_results=web_results,
            confidence=1.0  # High confidence for emergency protocols
        )
    
    def _should_search_web(self, analysis: Dict[str, Any], kb_results: list) -> bool:
        """Determine if web search is needed based on query analysis and KB results"""
        
        # Always search for emergency queries
        if analysis['is_emergency']:
            return True
        
        # Search if confidence is low
        if analysis['confidence'] < 0.6:
            return True
        
        # Search if knowledge base results are insufficient
        if len(kb_results) < 2:
            return True
        
        # Search if KB results have low confidence
        if kb_results and all(result.confidence < 0.7 for result in kb_results):
            return True
        
        # Don't search for high-confidence queries with good KB results
        return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            kb_stats = self.knowledge_retriever.get_knowledge_base_stats()
            
            return {
                'status': 'healthy',
                'components': {
                    'query_analyzer': 'active',
                    'knowledge_retriever': 'active',
                    'web_search_agent': 'active',
                    'response_generator': 'active'
                },
                'knowledge_base': kb_stats,
                'capabilities': [
                    'intent_classification',
                    'entity_extraction',
                    'semantic_search',
                    'web_search_integration',
                    'emergency_detection',
                    'response_generation',
                    'confidence_scoring',
                    'escalation_management'
                ],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                'status': 'degraded',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global orchestrator instance
rag_orchestrator = RAGOrchestrator()
