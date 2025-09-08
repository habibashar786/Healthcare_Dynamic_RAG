#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Healthcare RAG Orchestrator - With Pre-Authorization Support
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import httpx
import asyncio
from datetime import datetime, timedelta
import sys
import os
import json
import traceback
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

app = FastAPI(
    title="Healthcare RAG Orchestrator",
    description="Unified healthcare query system with pre-authorization support",
    version="3.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
INSURANCE_SERVER = "http://localhost:8002"
HOSPITAL_SERVER = "http://localhost:8003"

# Global data storage
GLOBAL_DATA = {
    "patients": [],
    "insurance": [],
    "cross_reference": [],
    "coverage_matrix": [],
    "procedures": [],
    "preauth_rules": [],
    "loaded": False
}

def clean_nan_values(data):
    """Clean NaN values from a dictionary or list for JSON compatibility"""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if pd.isna(value):
                cleaned[key] = None
            elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                cleaned[key] = None
            else:
                cleaned[key] = value
        return cleaned
    elif isinstance(data, list):
        return [clean_nan_values(item) for item in data]
    else:
        return data

@app.on_event("startup")
async def startup_event():
    """Initialize knowledge base with all regulatory data"""
    print("=" * 60)
    print("Initializing Healthcare RAG Orchestrator v3.4")
    print("=" * 60)
    
    # Load patient data
    try:
        patient_df = pd.read_csv("./data/patient_augmented_synthetic.csv")
        GLOBAL_DATA["patients"] = [clean_nan_values(record) for record in patient_df.to_dict('records')]
        print(f"✓ Loaded {len(GLOBAL_DATA['patients'])} patient records")
    except Exception as e:
        print(f"✗ Error loading patients: {e}")
        GLOBAL_DATA["patients"] = []
    
    # Load insurance data
    try:
        insurance_df = pd.read_csv("./data/insurance_policies_gcc.csv")
        GLOBAL_DATA["insurance"] = [clean_nan_values(record) for record in insurance_df.to_dict('records')]
        print(f"✓ Loaded {len(GLOBAL_DATA['insurance'])} insurance policies")
    except Exception as e:
        print(f"✗ Error loading insurance: {e}")
        GLOBAL_DATA["insurance"] = []
    
    # Load cross-reference data
    try:
        cross_ref_df = pd.read_csv("./data/cross_reference_table.csv")
        GLOBAL_DATA["cross_reference"] = [clean_nan_values(record) for record in cross_ref_df.to_dict('records')]
        print(f"✓ Loaded {len(GLOBAL_DATA['cross_reference'])} cross-references")
    except Exception as e:
        print(f"✗ Error loading cross-references: {e}")
        GLOBAL_DATA["cross_reference"] = []
    
    # Load coverage matrix
    try:
        coverage_df = pd.read_csv("./data/insurance_coverage_matrix.csv")
        GLOBAL_DATA["coverage_matrix"] = [clean_nan_values(record) for record in coverage_df.to_dict('records')]
        print(f"✓ Loaded {len(GLOBAL_DATA['coverage_matrix'])} coverage rules")
    except Exception as e:
        print(f"✗ Error loading coverage matrix: {e}")
        GLOBAL_DATA["coverage_matrix"] = []
    
    # Load procedures
    try:
        procedures_df = pd.read_csv("./data/specialty_procedures.csv")
        GLOBAL_DATA["procedures"] = [clean_nan_values(record) for record in procedures_df.to_dict('records')]
        print(f"✓ Loaded {len(GLOBAL_DATA['procedures'])} procedures")
    except Exception as e:
        print(f"✗ Error loading procedures: {e}")
        GLOBAL_DATA["procedures"] = []
    
    # Load pre-authorization rules
    try:
        preauth_df = pd.read_csv("./data/preauthorization_rules.csv")
        GLOBAL_DATA["preauth_rules"] = [clean_nan_values(record) for record in preauth_df.to_dict('records')]
        print(f"✓ Loaded {len(GLOBAL_DATA['preauth_rules'])} pre-authorization rules")
    except Exception as e:
        print(f"✗ Error loading pre-authorization rules: {e}")
        GLOBAL_DATA["preauth_rules"] = []
    
    GLOBAL_DATA["loaded"] = True
    print("=" * 60)
    print("RAG Orchestrator Ready!")
    print("=" * 60)

def extract_preauthorization_info(query: str, specialty: str = None) -> Dict:
    """Extract pre-authorization information and requirements"""
    preauth_info = {
        "process_overview": {},
        "requirements_by_tier": {},
        "common_procedures": [],
        "approval_timeline": {}
    }
    
    try:
        # General pre-authorization process
        preauth_info["process_overview"] = {
            "description": "Pre-authorization is required for certain medical procedures to ensure coverage",
            "steps": [
                "1. Doctor submits pre-authorization request with medical necessity",
                "2. Insurance reviews medical documentation",
                "3. Approval/denial decision within 24-72 hours",
                "4. Patient and provider notified of decision",
                "5. If approved, authorization number provided for billing"
            ]
        }
        
        # Requirements by insurance tier
        tier_requirements = {
            "Premium": {
                "requires_preauth": "Rarely - only for major surgeries",
                "approval_rate": "95%",
                "processing_time": "2-4 hours",
                "documentation": ["Doctor's recommendation"],
                "covered_without_preauth": [
                    "All consultations",
                    "Diagnostic tests",
                    "Emergency care",
                    "Preventive care",
                    "Most medications"
                ]
            },
            "Average": {
                "requires_preauth": "For specialized procedures",
                "approval_rate": "80%",
                "processing_time": "24-48 hours",
                "documentation": [
                    "Doctor's recommendation",
                    "Medical history",
                    "Test results"
                ],
                "covered_without_preauth": [
                    "Primary care visits",
                    "Basic diagnostic tests",
                    "Emergency care",
                    "Generic medications"
                ]
            },
            "Basic": {
                "requires_preauth": "For all non-emergency procedures",
                "approval_rate": "60%",
                "processing_time": "48-72 hours",
                "documentation": [
                    "Doctor's detailed recommendation",
                    "Complete medical history",
                    "All test results",
                    "Cost estimates"
                ],
                "covered_without_preauth": [
                    "Emergency care only",
                    "Basic consultations with referral"
                ]
            }
        }
        
        preauth_info["requirements_by_tier"] = tier_requirements
        
        # Common procedures requiring pre-authorization
        if GLOBAL_DATA["procedures"]:
            procedures_needing_auth = []
            
            # Filter procedures that typically need pre-auth
            high_cost_procedures = [
                p for p in GLOBAL_DATA["procedures"]
                if p.get("base_cost", 0) > 500
            ]
            
            for proc in high_cost_procedures[:10]:
                procedures_needing_auth.append({
                    "procedure": proc.get("procedure_name", "Unknown"),
                    "specialty": proc.get("specialty", "Unknown"),
                    "typical_cost": proc.get("base_cost", 0),
                    "preauth_required": {
                        "Premium": "No" if proc.get("base_cost", 0) < 2000 else "Yes",
                        "Average": "Yes",
                        "Basic": "Yes"
                    },
                    "approval_time": {
                        "Premium": "2-4 hours",
                        "Average": "24 hours",
                        "Basic": "48-72 hours"
                    }
                })
            
            preauth_info["common_procedures"] = procedures_needing_auth
        
        # Specialty-specific pre-authorization
        if specialty:
            specialty_rules = [
                rule for rule in GLOBAL_DATA["preauth_rules"]
                if rule.get("specialty", "").lower() == specialty.lower()
            ]
            
            if specialty_rules:
                preauth_info["specialty_requirements"] = {
                    "specialty": specialty,
                    "total_rules": len(specialty_rules),
                    "tiers": {}
                }
                
                for tier in ["Premium", "Average", "Basic"]:
                    tier_rules = [r for r in specialty_rules if r.get("policy_tier") == tier]
                    if tier_rules:
                        rule = tier_rules[0]
                        preauth_info["specialty_requirements"]["tiers"][tier] = {
                            "processing_hours": rule.get("processing_time_hours", 24),
                            "documentation": rule.get("documentation_required", ["Medical records"]),
                            "approval_criteria": rule.get("approval_criteria", {})
                        }
        
        # Approval timeline
        preauth_info["approval_timeline"] = {
            "Emergency": "Immediate approval, documentation within 24 hours",
            "Urgent": "4-8 hours",
            "Standard": "24-48 hours",
            "Elective": "48-72 hours"
        }
        
    except Exception as e:
        print(f"Error extracting pre-authorization info: {e}")
        traceback.print_exc()
    
    return preauth_info

def extract_detailed_patient_data(query: str, limit: int = 5) -> List[Dict]:
    """Extract detailed patient records matching query"""
    query_lower = query.lower()
    matches = []
    
    try:
        for patient in GLOBAL_DATA["patients"]:
            try:
                patient_str = json.dumps(patient).lower()
            except:
                continue
            
            if any(term in patient_str for term in query_lower.split()):
                match = {
                    "patient_id": patient.get("patient_id", "Unknown") or "Unknown",
                    "gender": patient.get("gender", "Unknown") or "Unknown",
                    "age": int(patient.get("age", 0)) if patient.get("age") is not None else 0,
                    "city": patient.get("city", "Unknown") or "Unknown",
                    "country": patient.get("country", "Unknown") or "Unknown",
                    "chief_complaint": patient.get("chief_complaint", "No complaint recorded") or "No complaint recorded",
                    "diagnosis": patient.get("diagnosis_icd10am", "No diagnosis") or "No diagnosis",
                    "medications": patient.get("medications", "None") or "None",
                    "insurance_policy_id": patient.get("insurance_policy_id", "No insurance") or "No insurance",
                    "encounter_date": str(patient.get("encounter_date", "")) if patient.get("encounter_date") else "",
                    "allergies": patient.get("allergies", "None") or "None",
                    "smoking_status": patient.get("smoking_status", "Unknown") or "Unknown"
                }
                matches.append(match)
                
                if len(matches) >= limit:
                    break
                    
    except Exception as e:
        print(f"Error extracting patient data: {e}")
        traceback.print_exc()
    
    return matches

def extract_insurance_coverage_details(query: str, specialty: str = None) -> Dict:
    """Extract detailed insurance coverage information"""
    coverage_info = {
        "tier_details": {},
        "specialty_coverage": {},
        "benefits": []
    }
    
    try:
        tier_benefits = {
            "Premium": {
                "coverage_percentage": "90-100%",
                "annual_limit": "$1,000,000",
                "deductible": "$0-500",
                "preauth_required": "Rarely",
                "waiting_period": "None",
                "network": "All providers",
                "benefits": [
                    "No waiting period for pre-existing conditions",
                    "100% preventive care coverage",
                    "Global emergency coverage",
                    "Free annual health checkups",
                    "Dental and vision included"
                ]
            },
            "Average": {
                "coverage_percentage": "60-80%",
                "annual_limit": "$500,000",
                "deductible": "$500-1500",
                "preauth_required": "For specialist care",
                "waiting_period": "3-6 months",
                "network": "Preferred providers",
                "benefits": [
                    "Coverage for major medical expenses",
                    "Specialist consultations with referral",
                    "Emergency care coverage",
                    "Generic medication coverage",
                    "Partial dental coverage"
                ]
            },
            "Basic": {
                "coverage_percentage": "30-50%",
                "annual_limit": "$250,000",
                "deductible": "$1500-3000",
                "preauth_required": "All non-emergency",
                "waiting_period": "12 months",
                "network": "Limited providers",
                "benefits": [
                    "Emergency care only",
                    "Basic medication coverage",
                    "Annual limit on claims",
                    "In-network providers only",
                    "No dental or vision"
                ]
            }
        }
        
        if specialty and GLOBAL_DATA["coverage_matrix"]:
            specialty_coverage = [
                rule for rule in GLOBAL_DATA["coverage_matrix"]
                if rule.get("specialty", "").lower() == specialty.lower()
            ]
            
            for tier in ["Premium", "Average", "Basic"]:
                tier_rules = [r for r in specialty_coverage if r.get("policy_tier") == tier]
                if tier_rules:
                    avg_coverage = sum(r.get("coverage_percentage", 0) for r in tier_rules) / len(tier_rules)
                    coverage_info["specialty_coverage"][tier] = {
                        "average_coverage": f"{avg_coverage:.0f}%",
                        "procedures_covered": len(tier_rules),
                        "requires_preauth": any(r.get("requires_preauth", False) for r in tier_rules),
                        "max_claim": tier_rules[0].get("max_claim_amount", 0) if tier_rules else 0
                    }
        
        for tier_name, benefits in tier_benefits.items():
            if tier_name.lower() in query.lower() or "premium" in query.lower() or "all" in query.lower():
                coverage_info["tier_details"][tier_name] = benefits
                coverage_info["benefits"].extend(benefits["benefits"])
        
        if not coverage_info["tier_details"]:
            coverage_info["tier_details"]["Premium"] = tier_benefits["Premium"]
            coverage_info["benefits"] = tier_benefits["Premium"]["benefits"]
    
    except Exception as e:
        print(f"Error extracting insurance coverage: {e}")
        traceback.print_exc()
    
    return coverage_info

def extract_procedure_costs(specialty: str) -> List[Dict]:
    """Extract procedure costs for a specialty"""
    procedures = []
    
    try:
        specialty_procedures = [
            p for p in GLOBAL_DATA["procedures"]
            if p.get("specialty", "").lower() == specialty.lower()
        ]
        
        for proc in specialty_procedures[:5]:
            base_cost = proc.get("base_cost", 0)
            if base_cost is None or pd.isna(base_cost):
                base_cost = 0
                
            procedures.append({
                "procedure_name": proc.get("procedure_name", "Unknown") or "Unknown",
                "base_cost": float(base_cost),
                "icd10_code": proc.get("icd10_code", "") or "",
                "urgency_level": proc.get("urgency_level", "Routine") or "Routine",
                "premium_cost": float(base_cost) * 0.1,
                "average_cost": float(base_cost) * 0.3,
                "basic_cost": float(base_cost) * 0.6
            })
    except Exception as e:
        print(f"Error extracting procedure costs: {e}")
        traceback.print_exc()
    
    return procedures

async def get_live_appointments(specialty: str) -> List[Dict]:
    """Get live appointment data from hospital server"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{HOSPITAL_SERVER}/hospital/availability/{specialty}",
                timeout=5.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("slots"):
                    return data["slots"][:10]
    except Exception as e:
        print(f"Error fetching appointments: {e}")
    
    return [
        {
            "slot_id": f"SLOT-SAMPLE-{i}",
            "date": f"2024-01-{15+i}",
            "time": f"{9+i}:00",
            "doctor_name": f"Dr. {specialty} Specialist",
            "location_name": "Doha Medical Center",
            "available": True
        }
        for i in range(3)
    ]

async def process_preauthorization_request(patient_id: str, procedure: str, tier: str) -> Dict:
    """Process a pre-authorization request"""
    try:
        # Generate authorization details
        auth_number = f"AUTH-{datetime.now().strftime('%Y%m%d')}-{random.randint(10000, 99999)}"
        
        # Determine approval based on tier
        if tier == "Premium":
            status = "APPROVED"
            processing_time = "2 hours"
            requirements = ["Doctor's recommendation"]
        elif tier == "Average":
            status = "PENDING REVIEW"
            processing_time = "24 hours"
            requirements = ["Doctor's recommendation", "Medical history", "Test results"]
        else:
            status = "UNDER REVIEW"
            processing_time = "48-72 hours"
            requirements = ["Complete medical documentation", "Cost justification", "Alternative treatment options"]
        
        return {
            "authorization_number": auth_number,
            "status": status,
            "procedure": procedure,
            "patient_id": patient_id,
            "tier": tier,
            "processing_time": processing_time,
            "requirements": requirements,
            "expiry_date": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "next_steps": [
                "Submit required documentation",
                "Wait for review completion",
                "Receive approval notification",
                "Schedule procedure within 30 days of approval"
            ]
        }
    except Exception as e:
        print(f"Error processing pre-authorization: {e}")
        return {}

@app.get("/")
async def root():
    return {
        "service": "Healthcare RAG Orchestrator",
        "status": "operational",
        "version": "3.4.0",
        "data_loaded": {
            "patients": len(GLOBAL_DATA["patients"]),
            "insurance": len(GLOBAL_DATA["insurance"]),
            "cross_references": len(GLOBAL_DATA["cross_reference"]),
            "coverage_rules": len(GLOBAL_DATA["coverage_matrix"]),
            "procedures": len(GLOBAL_DATA["procedures"]),
            "preauth_rules": len(GLOBAL_DATA["preauth_rules"])
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "rag_orchestrator": "healthy",
        "timestamp": datetime.now().isoformat(),
        "patients_loaded": len(GLOBAL_DATA["patients"]),
        "insurance_policies": len(GLOBAL_DATA["insurance"]),
        "cross_references": len(GLOBAL_DATA["cross_reference"]),
        "coverage_rules": len(GLOBAL_DATA["coverage_matrix"]),
        "procedures": len(GLOBAL_DATA["procedures"]),
        "preauth_rules": len(GLOBAL_DATA["preauth_rules"])
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{INSURANCE_SERVER}/", timeout=2.0)
            health_status["insurance_server"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        health_status["insurance_server"] = "offline"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{HOSPITAL_SERVER}/", timeout=2.0)
            health_status["hospital_server"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        health_status["hospital_server"] = "offline"
    
    return health_status

@app.post("/query")
async def unified_query(request: dict):
    """Enhanced query processing with pre-authorization support"""
    try:
        query = request.get('query', '').lower()
        print(f"Processing query: {query}")
        
        response_parts = []
        context_data = {}
        
        # Detect query intent and specialty
        specialties = ["cardiology", "internal medicine", "urology", "pediatrics", "emergency", "ob/gyn"]
        detected_specialty = None
        for spec in specialties:
            if spec in query:
                detected_specialty = spec.title()
                break
        
        # CASE 1: Pre-Authorization Query (FIXED)
        if any(term in query for term in ["preauth", "pre-auth", "pre auth", "authorization", "approval", "approve"]):
            preauth_info = extract_preauthorization_info(query, detected_specialty)
            
            # Add overview to response
            response_parts.append("Pre-authorization ensures your medical procedures are covered by insurance")
            
            # Add tier-specific information
            if preauth_info["requirements_by_tier"]:
                for tier, requirements in preauth_info["requirements_by_tier"].items():
                    response_parts.append(
                        f"{tier} Tier: {requirements['requires_preauth']}, "
                        f"Approval rate: {requirements['approval_rate']}, "
                        f"Processing: {requirements['processing_time']}"
                    )
            
            context_data["preauthorization"] = {
                "process": preauth_info["process_overview"],
                "requirements": preauth_info["requirements_by_tier"],
                "common_procedures": preauth_info["common_procedures"][:5] if preauth_info["common_procedures"] else [],
                "timeline": preauth_info["approval_timeline"]
            }
            
            # If specific specialty mentioned, add specialty requirements
            if detected_specialty and preauth_info.get("specialty_requirements"):
                context_data["specialty_preauth"] = preauth_info["specialty_requirements"]
                response_parts.append(f"For {detected_specialty}: specific pre-authorization rules apply")
        
        # CASE 2: Insurance Coverage Query
        elif any(term in query for term in ["insurance", "coverage", "premium", "tier", "benefits", "deductible"]):
            coverage_info = extract_insurance_coverage_details(query, detected_specialty)
            
            if coverage_info["tier_details"]:
                for tier_name, details in coverage_info["tier_details"].items():
                    response_parts.append(
                        f"{tier_name} Tier: {details['coverage_percentage']} coverage, "
                        f"{details['annual_limit']} annual limit, {details['deductible']} deductible"
                    )
                
                context_data["coverage_details"] = {
                    "tier": list(coverage_info["tier_details"].keys())[0],
                    "coverage_percentage": 90 if "Premium" in coverage_info["tier_details"] else 70,
                    "patient_responsibility": 100 if "Premium" in coverage_info["tier_details"] else 300,
                    "insurance_payment": 900 if "Premium" in coverage_info["tier_details"] else 700,
                    "requires_preauth": False if "Premium" in coverage_info["tier_details"] else True,
                    "benefits": coverage_info["benefits"][:5]
                }
            
            if detected_specialty and coverage_info["specialty_coverage"]:
                context_data["specialty_coverage"] = coverage_info["specialty_coverage"]
                response_parts.append(f"For {detected_specialty}: Coverage varies by tier")
        
        # CASE 3: Appointment Booking Query
        elif any(term in query for term in ["appointment", "book", "available", "schedule", "this week", "today"]):
            if detected_specialty:
                appointments = await get_live_appointments(detected_specialty)
                if appointments:
                    response_parts.append(
                        f"Found {len(appointments)} available appointments for {detected_specialty}"
                    )
                    context_data["appointments"] = appointments[:5]
            else:
                all_appointments = []
                for spec in ["Cardiology", "Internal Medicine"]:
                    slots = await get_live_appointments(spec)
                    all_appointments.extend(slots[:2])
                
                if all_appointments:
                    response_parts.append(
                        f"Found {len(all_appointments)} available appointments across specialties"
                    )
                    context_data["appointments"] = all_appointments
        
        # CASE 4: Emergency Services Query
        elif "emergency" in query:
            response_parts.append(
                "Emergency services available 24/7 at all locations. "
                "No pre-authorization required for emergency care"
            )
            
            emergency_info = {
                "available": "24/7",
                "locations": ["Doha Medical Center", "Al Wakrah Hospital"],
                "wait_time": "Immediate for critical cases",
                "coverage": "100% for life-threatening emergencies",
                "preauth_required": "No - treated first, paperwork later"
            }
            context_data["emergency_info"] = emergency_info
        
        # CASE 5: Medical Condition Query
        elif any(condition in query for condition in ["diabetes", "hypertension", "asthma", "cardiac", "covid"]):
            condition_patients = extract_detailed_patient_data(query, limit=3)
            if condition_patients:
                response_parts.append(
                    f"Found {len(condition_patients)} patient records with related conditions"
                )
                context_data["patient_matches"] = condition_patients
        
        # CASE 6: Location-based Query
        elif any(location in query for location in ["qatar", "doha", "wakrah"]):
            location_patients = extract_detailed_patient_data(query, limit=5)
            if location_patients:
                response_parts.append(
                    f"Found {len(location_patients)} patient records in the specified location"
                )
                context_data["patient_matches"] = location_patients
        
        # CASE 7: Cost Estimate Query
        elif any(term in query for term in ["cost", "price", "how much", "expense"]):
            if detected_specialty:
                procedures = extract_procedure_costs(detected_specialty)
                if procedures:
                    response_parts.append(
                        f"Cost estimates for {detected_specialty} procedures vary by insurance tier"
                    )
                    context_data["procedure_costs"] = procedures
        
        # CASE 8: Gender-based Patient Search
        elif any(term in query for term in ["male", "female"]):
            gender_patients = extract_detailed_patient_data(query, limit=5)
            if gender_patients:
                response_parts.append(
                    f"Found {len(gender_patients)} patient records matching your criteria"
                )
                context_data["patient_matches"] = gender_patients
        
        # Generate final response
        if response_parts:
            answer = ". ".join(response_parts)
            
            if detected_specialty and not context_data.get("appointments"):
                answer += f". To book an appointment for {detected_specialty}, please specify your preferred date."
            
            if "premium" in query and not context_data.get("coverage_details"):
                answer += ". Premium tier offers the best coverage with minimal out-of-pocket costs."
        else:
            answer = (
                "I couldn't find specific information for your query. "
                "Please contact our 24/7 customer support at 111-222-333 for personalized assistance."
            )
            
            context_data["support_required"] = True
            context_data["support_number"] = "111-222-333"
            context_data["support_hours"] = "24/7"
        
        return {
            "answer": answer,
            "sources": ["Patient Database", "Insurance Matrix", "Hospital Network"] if context_data else [],
            "confidence": 0.95 if context_data else 0.3,
            "status": "success",
            "context": context_data
        }
        
    except Exception as e:
        print(f"Error in unified_query: {e}")
        traceback.print_exc()
        
        return {
            "answer": "I encountered an issue processing your request. Please contact support at 111-222-333.",
            "sources": [],
            "confidence": 0.0,
            "status": "error",
            "context": {"error": str(e), "support_number": "111-222-333"}
        }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Starting Healthcare RAG Orchestrator v3.4")
    print("With Pre-Authorization Support")
    print("=" * 60)
    print("Main Server: http://localhost:8001")
    print("API Docs: http://localhost:8001/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001)