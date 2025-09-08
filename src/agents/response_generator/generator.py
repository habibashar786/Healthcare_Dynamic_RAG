from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Any
import logging
from src.agents.knowledge_retrieval.retriever import RetrievalResult
from src.agents.web_search.search_agent import SearchResult
from datetime import datetime

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, api_key: str = None):
        self.llm = OpenAI(temperature=0.3, openai_api_key=api_key)
        self.response_chain = self._setup_response_chain()
        
    def _setup_response_chain(self):
        """Setup response generation chain"""
        template = """
        You are a helpful healthcare AI assistant. Generate a comprehensive, accurate response based on the following information:
        
        Original Query: {query}
        Query Intent: {intent}
        
        Knowledge Base Results:
        {kb_results}
        
        Web Search Results:
        {web_results}
        
        Guidelines:
        1. Provide accurate, helpful healthcare information
        2. Always include disclaimers for medical advice
        3. Be empathetic and professional
        4. If information is limited, be transparent about it
        5. Include relevant contact information when appropriate
        6. For emergencies, emphasize immediate professional care
        
        Healthcare Disclaimer: "This information is for educational purposes only and should not replace professional medical advice. Always consult with a healthcare provider for medical concerns."
        
        Customer Service: For personalized assistance, call 0800-123-456
        
        Response:"""
        
        return PromptTemplate(template=template, input_variables=[
            "query", "intent", "kb_results", "web_results"
        ])
    
    async def generate_response(
        self, 
        query: str, 
        intent: str, 
        kb_results: List[RetrievalResult], 
        web_results: List[SearchResult],
        confidence: float
    ) -> Dict[str, Any]:
        """Generate comprehensive response from all available data"""
        
        try:
            # Format knowledge base results
            kb_text = self._format_kb_results(kb_results)
            
            # Format web search results  
            web_text = self._format_web_results(web_results)
            
            # Handle emergency queries
            if intent == "emergency":
                return self._generate_emergency_response(query, kb_text, web_text)
            
            # Handle low confidence queries
            if confidence < 0.5:
                return self._generate_escalation_response(query, kb_text, web_text)
            
            # Generate standard response using LLM
            response_text = self.response_chain.format(
                query=query,
                intent=intent,
                kb_results=kb_text,
                web_results=web_text
            )
            
            # Use LLM to generate response
            llm_response = self.llm(response_text)
            
            # Calculate final confidence
            final_confidence = self._calculate_response_confidence(
                kb_results, web_results, confidence
            )
            
            return {
                "status": "success",
                "answer": llm_response.strip(),
                "confidence": final_confidence,
                "sources": self._compile_sources(kb_results, web_results),
                "escalated": False,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._generate_error_response(query)
    
    def _format_kb_results(self, results: List[RetrievalResult]) -> str:
        """Format knowledge base results for prompt"""
        if not results:
            return "No relevant information found in knowledge base."
        
        formatted = "Knowledge Base Information:\n"
        for i, result in enumerate(results[:5], 1):
            formatted += f"{i}. Source: {result.source_type}\n"
            formatted += f"   Content: {result.content[:300]}...\n"
            formatted += f"   Confidence: {result.confidence:.2f}\n\n"
        
        return formatted
    
    def _format_web_results(self, results: List[SearchResult]) -> str:
        """Format web search results for prompt"""
        if not results:
            return "No relevant web results found."
        
        formatted = "Web Search Results:\n"
        for i, result in enumerate(results[:3], 1):
            formatted += f"{i}. Title: {result.title}\n"
            formatted += f"   Source: {result.url}\n"
            formatted += f"   Summary: {result.snippet}\n"
            formatted += f"   Relevance: {result.relevance_score:.2f}\n\n"
        
        return formatted
    
    def _generate_emergency_response(self, query: str, kb_text: str, web_text: str) -> Dict[str, Any]:
        """Generate emergency response"""
        emergency_response = f"""
        íº¨ MEDICAL EMERGENCY DETECTED íº¨
        
        For immediate medical emergencies, please:
        1. Call 911 or your local emergency number immediately
        2. If unconscious or not breathing, begin CPR if trained
        3. Stay calm and follow emergency operator instructions
        
        Your query: "{query}"
        
        This appears to be a medical emergency. While I can provide some general information, 
        immediate professional medical attention is crucial.
        
        Emergency Hotline: 911
        Poison Control: 1-800-222-1222
        Customer Service: 0800-123-456
        
        âš ï¸ Do not delay seeking immediate medical care for emergency situations.
        
        Available Information:
        {kb_text[:200]}
        
        Please seek immediate professional medical attention.
        """
        
        return {
            "status": "emergency",
            "answer": emergency_response.strip(),
            "confidence": 1.0,
            "sources": [{"type": "emergency_protocol", "priority": "immediate_care"}],
            "escalated": True,
            "emergency_contacts": {
                "emergency": "911",
                "poison_control": "1-800-222-1222",
                "customer_service": "0800-123-456"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_escalation_response(self, query: str, kb_text: str, web_text: str) -> Dict[str, Any]:
        """Generate response for low-confidence queries requiring human assistance"""
        escalation_response = f"""
        I understand you're asking about: "{query}"
        
        While I have some information available, I'd recommend speaking with a healthcare 
        professional for personalized advice and accurate guidance specific to your situation.
        
        Available Information:
        {kb_text[:300] if kb_text else "Limited information available in our knowledge base."}
        
        For detailed assistance and personalized healthcare guidance:
        í³ž Customer Service: 0800-123-456
        í³§ Email: support@healthguard.ai
        íµ’ Hours: 24/7 for emergencies, 9 AM - 6 PM for general inquiries
        
        Our healthcare specialists can provide:
        - Detailed insurance plan comparisons
        - Personalized hospital recommendations
        - Specific medical condition guidance
        - Appointment scheduling assistance
        
        Reference ID: HG-{datetime.now().strftime('%Y%m%d%H%M%S')}
        
        Please don't hesitate to contact our customer service team for comprehensive assistance.
        """
        
        return {
            "status": "escalated",
            "answer": escalation_response.strip(),
            "confidence": 0.7,
            "sources": [{"type": "customer_service", "action": "escalated"}],
            "escalated": True,
            "customer_service_info": {
                "phone": "0800-123-456",
                "email": "support@healthguard.ai",
                "hours": "24/7 for emergencies, 9 AM - 6 PM for general inquiries"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_error_response(self, query: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            "status": "error",
            "answer": "I'm experiencing technical difficulties processing your request. Please contact our customer service team at 0800-123-456 for immediate assistance.",
            "confidence": 0.1,
            "sources": [],
            "escalated": True,
            "customer_service_info": {
                "phone": "0800-123-456",
                "reason": "technical_error"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_response_confidence(
        self, 
        kb_results: List[RetrievalResult], 
        web_results: List[SearchResult], 
        query_confidence: float
    ) -> float:
        """Calculate final response confidence"""
        base_confidence = query_confidence
        
        # Boost for knowledge base results
        if kb_results:
            kb_confidence = sum(r.confidence for r in kb_results[:3]) / min(len(kb_results), 3)
            base_confidence += kb_confidence * 0.3
        
        # Boost for web results
        if web_results:
            web_confidence = sum(r.relevance_score for r in web_results[:3]) / min(len(web_results), 3)
            base_confidence += web_confidence * 0.2
        
        return min(base_confidence, 0.95)
    
    def _compile_sources(
        self, 
        kb_results: List[RetrievalResult], 
        web_results: List[SearchResult]
    ) -> List[Dict[str, Any]]:
        """Compile all sources used in response"""
        sources = []
        
        for result in kb_results[:3]:
            sources.append({
                "type": "knowledge_base",
                "source": result.source_type,
                "confidence": result.confidence,
                "metadata": result.metadata
            })
        
        for result in web_results[:3]:
            sources.append({
                "type": "web_search",
                "title": result.title,
                "url": result.url,
                "relevance": result.relevance_score
            })
        
        return sources

# Global response generator instance
response_generator = ResponseGenerator()
