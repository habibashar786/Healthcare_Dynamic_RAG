#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Insurance Server with Regulatory Compliance
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import json
from datetime import datetime
import random

app = FastAPI(
    title="Insurance Regulatory Server",
    description="Insurance verification and authorization with regulatory compliance",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load regulatory data
try:
    coverage_matrix = pd.read_csv("data/insurance_coverage_matrix.csv")
    preauth_rules = pd.read_csv("data/preauthorization_rules.csv")
    cross_reference = pd.read_csv("data/cross_reference_table.csv")
    procedures = pd.read_csv("data/specialty_procedures.csv")
    authorities = pd.read_csv("data/regulatory_authorities.csv")
    print(f"✓ Loaded regulatory data: {len(coverage_matrix)} coverage rules")
except Exception as e:
    print(f"Warning: Could not load all regulatory data: {e}")
    coverage_matrix = pd.DataFrame()
    preauth_rules = pd.DataFrame()
    cross_reference = pd.DataFrame()

# Request/Response Models
class CoverageCheckRequest(BaseModel):
    patient_id: str
    policy_id: str
    specialty: str
    procedure_code: Optional[str] = None
    estimated_cost: Optional[float] = None

class CoverageResponse(BaseModel):
    covered: bool
    coverage_percentage: float
    patient_responsibility: float
    insurance_payment: float
    requires_preauth: bool
    preauth_status: Optional[str] = None
    tier: str
    message: str

class PreAuthRequest(BaseModel):
    patient_id: str
    policy_id: str
    specialty: str
    procedure_code: str
    medical_necessity: str
    doctor_notes: Optional[str] = None

class PreAuthResponse(BaseModel):
    authorization_number: str
    status: str  # APPROVED, DENIED, PENDING
    processing_time_hours: int
    requirements: List[str]
    expiry_date: str

@app.get("/")
async def root():
    return {
        "service": "Insurance Regulatory Server",
        "status": "operational",
        "regulations": len(authorities) if not authorities.empty else 0,
        "coverage_rules": len(coverage_matrix) if not coverage_matrix.empty else 0
    }

@app.post("/insurance/verify-coverage", response_model=CoverageResponse)
async def verify_coverage(request: CoverageCheckRequest):
    """Verify insurance coverage for a procedure"""
    try:
        # Find the patient's policy tier from cross-reference
        patient_ref = cross_reference[
            (cross_reference['patient_id'] == request.patient_id) & 
            (cross_reference['insurance_policy_id'] == request.policy_id)
        ]
        
        if patient_ref.empty:
            # Create default coverage
            tier = "Basic"
            coverage_pct = 30
        else:
            tier = patient_ref.iloc[0]['policy_tier']
            coverage_pct = patient_ref.iloc[0]['coverage_percentage']
        
        # Find specific coverage rules
        coverage_rule = coverage_matrix[
            (coverage_matrix['policy_tier'] == tier) & 
            (coverage_matrix['specialty'] == request.specialty)
        ]
        
        if not coverage_rule.empty:
            rule = coverage_rule.iloc[0]
            coverage_pct = rule['coverage_percentage']
            requires_preauth = rule['requires_preauth']
            max_claim = rule['max_claim_amount']
        else:
            # Default values
            requires_preauth = tier != "Premium"
            max_claim = 50000 if tier == "Basic" else 500000
        
        # Calculate costs
        estimated_cost = request.estimated_cost or 1000
        insurance_payment = min(estimated_cost * coverage_pct / 100, max_claim)
        patient_responsibility = estimated_cost - insurance_payment
        
        # Check pre-authorization status
        preauth_status = "NOT_REQUIRED"
        if requires_preauth:
            # Check if pre-auth exists
            existing_auth = cross_reference[
                (cross_reference['patient_id'] == request.patient_id) &
                (cross_reference['specialty'] == request.specialty)
            ]
            if not existing_auth.empty:
                preauth_status = existing_auth.iloc[0].get('preauth_status', 'REQUIRED')
            else:
                preauth_status = "REQUIRED"
        
        return CoverageResponse(
            covered=True,
            coverage_percentage=coverage_pct,
            patient_responsibility=patient_responsibility,
            insurance_payment=insurance_payment,
            requires_preauth=requires_preauth,
            preauth_status=preauth_status,
            tier=tier,
            message=f"{tier} tier: {coverage_pct}% coverage for {request.specialty}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insurance/preauthorization", response_model=PreAuthResponse)
async def request_preauthorization(request: PreAuthRequest):
    """Request pre-authorization for a procedure"""
    try:
        # Find pre-auth rules
        auth_rules = preauth_rules[
            (preauth_rules['specialty'] == request.specialty)
        ]
        
        if not auth_rules.empty:
            rule = auth_rules.iloc[0]
            processing_time = rule.get('processing_time_hours', 24)
            
            # Parse requirements
            try:
                requirements = json.loads(rule.get('documentation_required', '[]'))
            except:
                requirements = ["Medical records", "Doctor recommendation"]
        else:
            processing_time = 24
            requirements = ["Medical records", "Doctor recommendation"]
        
        # Simulate approval logic
        if "emergency" in request.medical_necessity.lower():
            status = "APPROVED"
            processing_time = 1
        elif "urgent" in request.medical_necessity.lower():
            status = "APPROVED"
            processing_time = 4
        else:
            # Random approval for demo (in production, this would be based on actual rules)
            status = random.choice(["APPROVED", "APPROVED", "PENDING"])  # 66% approval rate
            
        # Generate authorization number
        auth_number = f"AUTH-{datetime.now().strftime('%Y%m%d')}-{random.randint(10000, 99999)}"
        
        # Calculate expiry (30 days from now)
        expiry_date = (datetime.now() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        
        return PreAuthResponse(
            authorization_number=auth_number,
            status=status,
            processing_time_hours=processing_time,
            requirements=requirements,
            expiry_date=expiry_date
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/insurance/policy-details/{policy_id}")
async def get_policy_details(policy_id: str):
    """Get detailed policy information"""
    try:
        # Find policy in cross-reference
        policy_refs = cross_reference[cross_reference['insurance_policy_id'] == policy_id]
        
        if policy_refs.empty:
            raise HTTPException(status_code=404, detail="Policy not found")
        
        # Get unique tier
        tier = policy_refs.iloc[0]['policy_tier']
        
        # Get coverage details for all specialties
        coverage_details = {}
        for specialty in procedures['specialty'].unique():
            specialty_coverage = coverage_matrix[
                (coverage_matrix['policy_tier'] == tier) & 
                (coverage_matrix['specialty'] == specialty)
            ]
            
            if not specialty_coverage.empty:
                coverage_details[specialty] = {
                    "coverage_percentage": specialty_coverage.iloc[0]['coverage_percentage'],
                    "requires_preauth": bool(specialty_coverage.iloc[0]['requires_preauth']),
                    "waiting_period_days": int(specialty_coverage.iloc[0]['waiting_period_days'])
                }
        
        return {
            "policy_id": policy_id,
            "tier": tier,
            "coverage_by_specialty": coverage_details,
            "annual_limit": 1000000 if tier == "Premium" else 500000 if tier == "Average" else 250000,
            "deductible": 0 if tier == "Premium" else 1000 if tier == "Average" else 2000
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/insurance/coverage-matrix/{specialty}")
async def get_coverage_matrix(specialty: str):
    """Get coverage matrix for a specific specialty"""
    try:
        specialty_coverage = coverage_matrix[coverage_matrix['specialty'] == specialty]
        
        if specialty_coverage.empty:
            raise HTTPException(status_code=404, detail=f"No coverage found for {specialty}")
        
        # Group by tier
        result = {}
        for tier in ["Premium", "Average", "Basic"]:
            tier_coverage = specialty_coverage[specialty_coverage['policy_tier'] == tier]
            if not tier_coverage.empty:
                result[tier] = {
                    "coverage_percentage": tier_coverage['coverage_percentage'].mean(),
                    "requires_preauth": tier_coverage['requires_preauth'].any(),
                    "procedures_covered": len(tier_coverage)
                }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting Insurance Regulatory Server...")
    print("Server will run on: http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
