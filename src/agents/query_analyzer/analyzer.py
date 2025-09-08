from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
from typing import Dict, List, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class QueryIntent(str, Enum):
    HOSPITAL_SEARCH = "hospital_search"
    INSURANCE_INQUIRY = "insurance_inquiry"
    MEDICAL_CONDITION = "medical_condition"
    EMERGENCY = "emergency"
    GENERAL_HEALTH = "general_health"
    APPOINTMENT_BOOKING = "appointment_booking"
    MEDICATION_INFO = "medication_info"
    UNCLEAR = "unclear"

class QueryAnalyzer:
    def __init__(self, api_key: str = None):
        self.llm = OpenAI(temperature=0, openai_api_key=api_key)
        self.intent_classifier = self._setup_intent_classifier()
        self.entity_extractor = self._setup_entity_extractor()
        
    def _setup_intent_classifier(self):
        """Setup intent classification chain"""
        template = """
        Analyze the following healthcare query and classify its intent.
        
        Query: {query}
        
        Possible intents:
        - hospital_search: Looking for hospitals, doctors, or medical facilities
        - insurance_inquiry: Questions about insurance plans, coverage, costs
        - medical_condition: Questions about symptoms, diseases, treatments
        - emergency: Urgent medical situations requiring immediate attention
        - general_health: General health and wellness questions
        - appointment_booking: Scheduling medical appointments
        - medication_info: Questions about medications, dosages, side effects
        - unclear: Query intent is unclear or ambiguous
        
        Emergency keywords: emergency, urgent, chest pain, heart attack, stroke, bleeding, unconscious, severe pain, can't breathe
        
        Intent: """
        
        prompt = PromptTemplate(template=template, input_variables=["query"])
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def _setup_entity_extractor(self):
        """Setup entity extraction chain"""
        template = """
        Extract relevant entities from this healthcare query:
        
        Query: {query}
        
        Extract the following entities if present:
        - Medical specialties (cardiology, neurology, etc.)
        - Locations (cities, areas, addresses)
        - Medical conditions (diabetes, hypertension, etc.)
        - Insurance types (HMO, PPO, Bronze, etc.)
        - Age ranges or demographics
        - Urgency level (emergency, urgent, routine)
        
        Format as JSON:
        {{
            "specialties": [],
            "locations": [],
            "conditions": [],
            "insurance_types": [],
            "demographics": [],
            "urgency": "routine"
        }}
        
        Entities: """
        
        prompt = PromptTemplate(template=template, input_variables=["query"])
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query and extract intent and entities"""
        try:
            # Classify intent
            intent_result = self.intent_classifier.run(query=query).strip().lower()
            
            # Map to enum
            intent = QueryIntent.UNCLEAR
            for query_intent in QueryIntent:
                if query_intent.value in intent_result:
                    intent = query_intent
                    break
            
            # Check for emergency keywords
            emergency_keywords = [
                "emergency", "urgent", "chest pain", "heart attack", "stroke", 
                "bleeding", "unconscious", "severe pain", "can't breathe", "911"
            ]
            
            is_emergency = any(keyword in query.lower() for keyword in emergency_keywords)
            if is_emergency:
                intent = QueryIntent.EMERGENCY
            
            # Extract entities
            try:
                entities_result = self.entity_extractor.run(query=query)
                # Try to parse JSON, fallback to keyword extraction if failed
                try:
                    import json
                    entities = json.loads(entities_result)
                except:
                    entities = self._fallback_entity_extraction(query)
            except:
                entities = self._fallback_entity_extraction(query)
            
            # Calculate confidence based on keyword matches and clarity
            confidence = self._calculate_confidence(query, intent, entities)
            
            return {
                "intent": intent.value,
                "entities": entities,
                "confidence": confidence,
                "is_emergency": is_emergency,
                "original_query": query,
                "needs_escalation": confidence < 0.5 or is_emergency
            }
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return {
                "intent": QueryIntent.UNCLEAR.value,
                "entities": {},
                "confidence": 0.1,
                "is_emergency": False,
                "original_query": query,
                "needs_escalation": True
            }
    
    def _fallback_entity_extraction(self, query: str) -> Dict:
        """Fallback entity extraction using keyword matching"""
        query_lower = query.lower()
        
        # Medical specialties
        specialties = []
        specialty_keywords = {
            "cardiology": ["heart", "cardiac", "cardio"],
            "neurology": ["brain", "neuro", "nervous"],
            "orthopedics": ["bone", "joint", "ortho"],
            "pediatrics": ["child", "kids", "pediatric"],
            "oncology": ["cancer", "tumor", "oncology"],
            "dermatology": ["skin", "derma"],
            "psychiatry": ["mental", "depression", "anxiety"]
        }
        
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in query_lower for keyword in keywords + [specialty]):
                specialties.append(specialty)
        
        # Insurance types
        insurance_types = []
        insurance_keywords = ["hmo", "ppo", "bronze", "silver", "gold", "platinum", "medicare", "medicaid"]
        for ins_type in insurance_keywords:
            if ins_type in query_lower:
                insurance_types.append(ins_type)
        
        # Common conditions
        conditions = []
        condition_keywords = {
            "diabetes": ["diabetes", "diabetic", "blood sugar"],
            "hypertension": ["blood pressure", "hypertension", "high pressure"],
            "asthma": ["asthma", "breathing", "inhaler"],
            "arthritis": ["arthritis", "joint pain"]
        }
        
        for condition, keywords in condition_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                conditions.append(condition)
        
        return {
            "specialties": specialties,
            "locations": [],
            "conditions": conditions,
            "insurance_types": insurance_types,
            "demographics": [],
            "urgency": "emergency" if any(word in query_lower for word in ["emergency", "urgent"]) else "routine"
        }
    
    def _calculate_confidence(self, query: str, intent: QueryIntent, entities: Dict) -> float:
        """Calculate confidence score for query analysis"""
        base_confidence = 0.5
        
        # Boost confidence for clear keywords
        if intent != QueryIntent.UNCLEAR:
            base_confidence += 0.2
        
        # Boost for extracted entities
        entity_count = sum(len(v) if isinstance(v, list) else 1 for v in entities.values() if v)
        base_confidence += min(entity_count * 0.1, 0.3)
        
        # Reduce confidence for very short or unclear queries
        if len(query.split()) < 3:
            base_confidence -= 0.2
        
        return min(max(base_confidence, 0.1), 0.95)

# Example usage
if __name__ == "__main__":
    analyzer = QueryAnalyzer()
    
    test_queries = [
        "Find hospitals with cardiology department near downtown",
        "What insurance plans cover diabetes medication?",
        "Emergency! Chest pain and shortness of breath",
        "I need affordable health insurance"
    ]
    
    for query in test_queries:
        result = analyzer.analyze_query(query)
        print(f"Query: {query}")
        print(f"Result: {result}")
        print("-" * 50)
