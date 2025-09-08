from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"

class PatientRecord(BaseModel):
    patient_id: str
    gender: Gender
    age: int
    date_of_birth: str
    country: str
    city: str
    encounter_id: str
    encounter_date: str
    chief_complaint: str
    diagnosis_icd10am: str
    procedure_achi: Optional[str] = None
    medications: Optional[str] = None
    allergies: Optional[str] = None
    smoking_status: Optional[str] = None
    height_cm: Optional[int] = None
    weight_kg: Optional[int] = None
    bmi: Optional[float] = None
    bp_systolic: Optional[int] = None
    bp_diastolic: Optional[int] = None
    heart_rate_bpm: Optional[int] = None
    temp_c: Optional[float] = None
    respiratory_rate: Optional[int] = None
    spo2_percent: Optional[int] = None
    lab_hba1c_percent: Optional[float] = None
    lab_ldl_mgdl: Optional[int] = None
    lab_creatinine_mgdl: Optional[float] = None
    insurance_policy_id: str
    provider_id: str
    facility_id: str
    data_source: str

class InsurancePolicy(BaseModel):
    policy_id: str
    country: str
    regulator: str
    payer_name: str
    plan_tier: str
    currency: str
    annual_coverage_limit_local: int
    deductible_local: int
    copay_percent: int
    network: str
    outpatient_coverage: str
    inpatient_coverage: str
    pharmacy_coverage: str
    maternity_waiting_period_months: int
    preexisting_waiting_period_months: int
    cross_border_care: str
    phi_encrypted_at_rest: str
    phi_encrypted_in_transit: str
    access_controls_role_based: str
    audit_logs_enabled: str
    breach_notification_window_days: int
    data_retention_years: int
    minimum_necessary_policy: str
    business_associate_agreements: str
    eclaims_standard: str
    notes: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
class AgentResponse(BaseModel):
    agent_id: str
    task_id: str
    status: str
    confidence: float
    data: dict
    timestamp: datetime = Field(default_factory=datetime.now)
    
class RAGResponse(BaseModel):
    status: str
    answer: str
    confidence: float
    sources: List[dict]
    escalated: bool = False
    customer_service_info: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.now)
