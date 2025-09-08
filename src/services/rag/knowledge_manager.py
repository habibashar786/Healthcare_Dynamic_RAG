import pandas as pd
import pinecone
from typing import List, Dict, Any, Optional
import logging
import os
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class PineconeHealthcareKB:
    def __init__(self):
        # Initialize Pinecone (will need API key)
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if self.pinecone_api_key:
            try:
                pinecone.init(api_key=self.pinecone_api_key, environment="us-west1-gcp")
                self.index_name = "healthcare-rag"
                
                # Create index if it doesn't exist
                if self.index_name not in pinecone.list_indexes():
                    pinecone.create_index(
                        name=self.index_name,
                        dimension=384,  # MiniLM embedding dimension
                        metric='cosine'
                    )
                
                self.index = pinecone.Index(self.index_name)
                self.pinecone_available = True
                logger.info("Pinecone initialized successfully")
                
            except Exception as e:
                logger.error(f"Pinecone initialization failed: {str(e)}")
                self.pinecone_available = False
        else:
            logger.warning("No Pinecone API key found")
            self.pinecone_available = False
            
        # Fallback data storage
        self.patients_data = []
        self.insurance_data = []
    
    def load_patient_data(self, csv_path: str) -> bool:
        """Load patient data and create embeddings"""
        try:
            df = pd.read_csv(csv_path)
            self.patients_data = df.to_dict('records')
            
            if self.pinecone_available:
                return self._create_patient_embeddings(df)
            
            logger.info(f"Loaded {len(self.patients_data)} patient records (text-based)")
            return True
            
        except Exception as e:
            logger.error(f"Error loading patient data: {str(e)}")
            return False
    
    def _create_patient_embeddings(self, df: pd.DataFrame) -> bool:
        """Create embeddings for patient data"""
        try:
            vectors = []
            for idx, row in df.iterrows():
                # Create text representation
                text = f"""
                Patient: {row['age']} year old {row['gender']} from {row['city']}
                Chief Complaint: {row['chief_complaint']}
                Diagnosis: {row['diagnosis_icd10am']}
                """
                
                # Generate embedding
                embedding = self.embedding_model.encode(text.strip()).tolist()
                
                vectors.append({
                    'id': f"patient_{idx}",
                    'values': embedding,
                    'metadata': {
                        'type': 'patient',
                        'age': int(row['age']),
                        'gender': row['gender'],
                        'city': row['city'],
                        'chief_complaint': row['chief_complaint'],
                        'diagnosis': row['diagnosis_icd10am']
                    }
                })
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)
            logger.info(f"Created embeddings for {len(vectors)} patient records")
            return True
            
        except Exception as e:
            logger.error(f"Error creating patient embeddings: {str(e)}")
            return False

# Test the installation
pinecone_kb = PineconeHealthcareKB()
