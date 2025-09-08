from src.services.rag.knowledge_manager import pinecone_kb
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    source_type: str
    content: str
    confidence: float
    metadata: Dict[str, Any]

class KnowledgeRetriever:
    def __init__(self):
        self.kb = pinecone_kb
        
    async def retrieve_relevant_data(self, query: str, intent: str, entities: Dict) -> List[RetrievalResult]:
        """Retrieve relevant data using Pinecone or fallback search"""
        results = []
        
        try:
            if intent == "hospital_search":
                results.extend(await self._search_hospital_data(query, entities))
            elif intent == "insurance_inquiry":
                results.extend(await self._search_insurance_data(query, entities))
            elif intent == "medical_condition":
                results.extend(await self._search_medical_conditions(query, entities))
            elif intent in ["general_health", "medication_info"]:
                results.extend(await self._search_all_collections(query))
                
            results.sort(key=lambda x: x.confidence, reverse=True)
            return results[:10]
            
        except Exception as e:
            logger.error(f"Error retrieving data: {str(e)}")
            return []
    
    async def _search_hospital_data(self, query: str, entities: Dict) -> List[RetrievalResult]:
        """Search hospital/patient data"""
        results = []
        
        if self.kb.pinecone_available:
            # Use Pinecone semantic search
            try:
                query_embedding = self.kb.embedding_model.encode(query).tolist()
                search_results = self.kb.index.query(
                    vector=query_embedding,
                    filter={"type": "patient"},
                    top_k=5,
                    include_metadata=True
                )
                
                for match in search_results['matches']:
                    metadata = match['metadata']
                    content = f"""
                    Patient: {metadata.get('age')} year old {metadata.get('gender')} from {metadata.get('city')}
                    Chief Complaint: {metadata.get('chief_complaint')}
                    Diagnosis: {metadata.get('diagnosis')}
                    """
                    
                    results.append(RetrievalResult(
                        source_type="hospital",
                        content=content.strip(),
                        confidence=match['score'],
                        metadata=metadata
                    ))
                    
            except Exception as e:
                logger.error(f"Pinecone search error: {str(e)}")
        else:
            # Fallback to text search
            results.extend(self._fallback_patient_search(query, entities))
        
        return results
    
    def _fallback_patient_search(self, query: str, entities: Dict) -> List[RetrievalResult]:
        """Fallback text-based patient search"""
        results = []
        query_lower = query.lower()
        
        for patient in self.kb.patients_data[:10]:  # Limit search
            score = 0
            if query_lower in str(patient.get('chief_complaint', '')).lower():
                score += 0.8
            if query_lower in str(patient.get('diagnosis_icd10am', '')).lower():
                score += 0.7
                
            if score > 0:
                content = f"""
                Patient: {patient.get('age')} year old {patient.get('gender')} from {patient.get('city')}
                Chief Complaint: {patient.get('chief_complaint')}
                Diagnosis: {patient.get('diagnosis_icd10am')}
                """
                
                results.append(RetrievalResult(
                    source_type="hospital",
                    content=content.strip(),
                    confidence=score,
                    metadata=patient
                ))
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)[:5]
    
    async def _search_insurance_data(self, query: str, entities: Dict) -> List[RetrievalResult]:
        """Search insurance data (text-based for now)"""
        results = []
        query_lower = query.lower()
        
        for policy in self.kb.insurance_data[:20]:
            score = 0
            if query_lower in str(policy.get('payer_name', '')).lower():
                score += 0.7
            if 'affordable' in query_lower and policy.get('monthly_premium', 1000) < 400:
                score += 0.8
                
            if score > 0 or 'insurance' in query_lower:
                content = f"""
                Insurance: {policy.get('payer_name')} - {policy.get('plan_tier')}
                Premium: {policy.get('monthly_premium')} {policy.get('currency')}/month
                Coverage: {policy.get('annual_coverage_limit_local')} {policy.get('currency')}
                """
                
                results.append(RetrievalResult(
                    source_type="insurance",
                    content=content.strip(),
                    confidence=score if score > 0 else 0.3,
                    metadata=policy
                ))
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)[:5]
    
    async def _search_medical_conditions(self, query: str, entities: Dict) -> List[RetrievalResult]:
        """Search medical conditions"""
        return await self._search_hospital_data(query, entities)
    
    async def _search_all_collections(self, query: str) -> List[RetrievalResult]:
        """Search all collections"""
        results = []
        results.extend(self._fallback_patient_search(query, {}))
        results.extend(await self._search_insurance_data(query, {}))
        return results
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get KB statistics"""
        return {
            "collections": {
                "patients": len(self.kb.patients_data),
                "insurance_policies": len(self.kb.insurance_data),
                "medical_conditions": 0
            },
            "pinecone_available": self.kb.pinecone_available,
            "status": "active"
        }

knowledge_retriever = KnowledgeRetriever()
