#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hospital Server with Appointment Booking and Availability
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
from datetime import datetime, timedelta
import random

app = FastAPI(
    title="Hospital Services Server",
    description="Hospital appointment booking and procedure management",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data
try:
    procedures = pd.read_csv("data/specialty_procedures.csv")
    print(f"✓ Loaded {len(procedures)} procedures")
except:
    procedures = pd.DataFrame()

# Hospital Locations
LOCATIONS = {
    "LOC001": {
        "name": "Doha Medical Center",
        "address": "123 Al Sadd Street, Doha, Qatar",
        "specialties": ["Internal Medicine", "Cardiology", "Emergency"],
        "coordinates": {"lat": 25.2854, "lng": 51.5310}
    },
    "LOC002": {
        "name": "West Bay Clinic",
        "address": "456 Pearl Boulevard, West Bay, Qatar",
        "specialties": ["Pediatrics", "OB/GYN", "Dermatology"],
        "coordinates": {"lat": 25.3215, "lng": 51.5305}
    },
    "LOC003": {
        "name": "Al Wakrah Hospital",
        "address": "789 Coastal Road, Al Wakrah, Qatar",
        "specialties": ["Urology", "Orthopedics", "Neurology", "Emergency"],
        "coordinates": {"lat": 25.1717, "lng": 51.6067}
    }
}

# Request/Response Models
class AvailabilityRequest(BaseModel):
    specialty: str
    location_id: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None

class AppointmentSlot(BaseModel):
    slot_id: str
    date: str
    time: str
    doctor_name: str
    location_id: str
    location_name: str
    available: bool

class BookingRequest(BaseModel):
    patient_id: str
    slot_id: str
    specialty: str
    location_id: str
    insurance_auth_number: Optional[str] = None
    reason: Optional[str] = None

class BookingResponse(BaseModel):
    booking_id: str
    status: str
    appointment_date: str
    appointment_time: str
    doctor_name: str
    location: str
    estimated_wait: int
    preparation_instructions: List[str]

class CostEstimateRequest(BaseModel):
    specialty: str
    procedure_name: Optional[str] = None
    insurance_tier: Optional[str] = "Basic"

@app.get("/")
async def root():
    return {
        "service": "Hospital Services Server",
        "status": "operational",
        "locations": len(LOCATIONS),
        "procedures": len(procedures) if not procedures.empty else 0
    }

@app.get("/hospital/availability/{specialty}")
async def get_availability(specialty: str, location_id: Optional[str] = None):
    """Get available appointment slots for a specialty"""
    try:
        # Filter locations by specialty
        available_locations = []
        for loc_id, loc_info in LOCATIONS.items():
            if specialty in loc_info["specialties"]:
                if location_id is None or location_id == loc_id:
                    available_locations.append((loc_id, loc_info))
        
        if not available_locations:
            raise HTTPException(status_code=404, detail=f"No locations found for {specialty}")
        
        # Generate available slots
        slots = []
        doctors = {
            "Internal Medicine": ["Dr. Ahmed Hassan", "Dr. Sarah Johnson", "Dr. Maria Garcia"],
            "Cardiology": ["Dr. John Smith", "Dr. Fatima Al-Rashid", "Dr. Chen Wei"],
            "Urology": ["Dr. Robert Brown", "Dr. Ali Mohammed", "Dr. Kumar Patel"],
            "Pediatrics": ["Dr. Emily White", "Dr. Layla Ibrahim", "Dr. Jose Martinez"],
            "OB/GYN": ["Dr. Aisha Khan", "Dr. Rachel Green", "Dr. Noor Al-Sayed"],
            "Emergency": ["Dr. Emergency Team A", "Dr. Emergency Team B"],
        }
        
        specialty_doctors = doctors.get(specialty, ["Dr. Specialist"])
        
        # Generate slots for next 7 days
        for days_ahead in range(1, 8):
            date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            
            for loc_id, loc_info in available_locations:
                for hour in [9, 10, 11, 14, 15, 16]:  # Morning and afternoon slots
                    if random.random() > 0.3:  # 70% chance of availability
                        slot_id = f"SLOT-{date}-{hour:02d}-{loc_id[-3:]}"
                        slots.append(AppointmentSlot(
                            slot_id=slot_id,
                            date=date,
                            time=f"{hour:02d}:00",
                            doctor_name=random.choice(specialty_doctors),
                            location_id=loc_id,
                            location_name=loc_info["name"],
                            available=True
                        ))
        
        return {
            "specialty": specialty,
            "total_slots": len(slots),
            "slots": slots[:20]  # Return first 20 slots
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hospital/appointment/book", response_model=BookingResponse)
async def book_appointment(request: BookingRequest):
    """Book an appointment"""
    try:
        # Validate location
        if request.location_id not in LOCATIONS:
            raise HTTPException(status_code=404, detail="Invalid location")
        
        location = LOCATIONS[request.location_id]
        
        # Extract date and time from slot_id
        # Format: SLOT-YYYY-MM-DD-HH-XXX
        parts = request.slot_id.split("-")
        if len(parts) >= 4:
            date_str = f"{parts[1]}-{parts[2]}-{parts[3]}"
            time_str = f"{parts[4]}:00"
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = "10:00"
        
        # Generate booking confirmation
        booking_id = f"BOOK-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}"
        
        # Prepare instructions based on specialty
        instructions = {
            "Cardiology": ["Fast for 8 hours before appointment", "Bring previous ECG reports"],
            "Urology": ["Drink plenty of water before appointment", "Bring urine sample"],
            "Pediatrics": ["Bring vaccination records", "Child should be well-rested"],
            "OB/GYN": ["Bring previous ultrasound reports", "Wear comfortable clothing"],
            "Internal Medicine": ["Bring list of current medications", "Fast if blood work required"],
            "Emergency": ["Come immediately", "Bring insurance card and ID"]
        }
        
        return BookingResponse(
            booking_id=booking_id,
            status="CONFIRMED",
            appointment_date=date_str,
            appointment_time=time_str,
            doctor_name=f"Dr. {request.specialty} Specialist",
            location=location["name"],
            estimated_wait=random.randint(5, 30),
            preparation_instructions=instructions.get(request.specialty, ["No special preparation required"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hospital/procedures/{specialty}")
async def get_procedures(specialty: str):
    """Get available procedures for a specialty"""
    try:
        if procedures.empty:
            # Return sample data if procedures not loaded
            return {
                "specialty": specialty,
                "procedures": [
                    {"name": "Consultation", "code": "99213", "cost": 150},
                    {"name": "Follow-up", "code": "99214", "cost": 100}
                ]
            }
        
        specialty_procs = procedures[procedures['specialty'] == specialty]
        
        if specialty_procs.empty:
            raise HTTPException(status_code=404, detail=f"No procedures found for {specialty}")
        
        result = []
        for _, proc in specialty_procs.iterrows():
            result.append({
                "procedure_id": proc['procedure_id'],
                "name": proc['procedure_name'],
                "icd10_code": proc['icd10_code'],
                "cpt_code": proc['cpt_code'],
                "base_cost": proc['base_cost'],
                "urgency": proc['urgency_level']
            })
        
        return {
            "specialty": specialty,
            "total_procedures": len(result),
            "procedures": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hospital/cost-estimate")
async def estimate_cost(request: CostEstimateRequest):
    """Estimate procedure cost"""
    try:
        # Find procedure
        if not procedures.empty and request.procedure_name:
            proc = procedures[
                (procedures['specialty'] == request.specialty) & 
                (procedures['procedure_name'].str.contains(request.procedure_name, case=False))
            ]
            
            if not proc.empty:
                base_cost = proc.iloc[0]['base_cost']
            else:
                base_cost = 500  # Default cost
        else:
            base_cost = 500  # Default cost
        
        # Calculate based on tier
        tier_multiplier = {
            "Premium": 0.1,  # Patient pays 10%
            "Average": 0.3,  # Patient pays 30%
            "Basic": 0.6     # Patient pays 60%
        }
        
        patient_cost = base_cost * tier_multiplier.get(request.insurance_tier, 1.0)
        insurance_covers = base_cost - patient_cost
        
        return {
            "specialty": request.specialty,
            "procedure": request.procedure_name or "General Consultation",
            "total_cost": base_cost,
            "insurance_covers": insurance_covers,
            "patient_pays": patient_cost,
            "insurance_tier": request.insurance_tier,
            "currency": "QAR"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting Hospital Services Server...")
    print("Server will run on: http://localhost:8003")
    uvicorn.run(app, host="0.0.0.0", port=8003)
