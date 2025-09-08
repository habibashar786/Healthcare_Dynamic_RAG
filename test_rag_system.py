#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive RAG System Test Script
"""
import requests
import json
from datetime import datetime

# API Configuration
API_URL = "http://localhost:8001"

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_info(text):
    print(f"{YELLOW}ℹ {text}{RESET}")

def test_health():
    """Test health endpoint"""
    print_header("Testing Health Check")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Server is healthy")
            print_info(f"Patients loaded: {data.get('patients_loaded', 0)}")
            print_info(f"Insurance policies: {data.get('insurance_policies', 0)}")
            print_info(f"Cross-references: {data.get('cross_references', 0)}")
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Cannot connect to server: {e}")
        return False

def test_query(query_text, expected_keywords):
    """Test a specific query"""
    print(f"\nQuery: '{query_text}'")
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query_text},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('answer', '')
            context = data.get('context', {})
            
            # Check if expected keywords are in answer
            found_keywords = any(keyword.lower() in answer.lower() for keyword in expected_keywords)
            
            if found_keywords:
                print_success(f"Query successful")
                print(f"  Answer: {answer[:100]}...")
                
                # Display context details
                if context:
                    if 'patient_matches' in context:
                        print_info(f"  Found {len(context['patient_matches'])} patient records")
                    if 'appointments' in context:
                        print_info(f"  Found {len(context['appointments'])} appointment slots")
                    if 'coverage_details' in context:
                        print_info(f"  Coverage tier: {context['coverage_details'].get('tier', 'N/A')}")
                return True
            else:
                print_error(f"Query response missing expected content")
                print(f"  Expected keywords: {expected_keywords}")
                print(f"  Got: {answer[:100]}...")
                return False
        else:
            print_error(f"Query failed: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Query error: {e}")
        return False

def main():
    """Run all tests"""
    print_header("Healthcare RAG System Test Suite")
    print(f"Testing server at: {API_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if server is running
    if not test_health():
        print_error("Server is not running! Start it with: python src/api/main.py")
        return
    
    # Test queries
    test_cases = [
        {
            "query": "What is premium insurance coverage?",
            "keywords": ["Premium", "coverage", "90", "100", "deductible"]
        },
        {
            "query": "Show available appointments for cardiology",
            "keywords": ["appointment", "cardiology", "available", "found"]
        },
        {
            "query": "Find patients in Qatar",
            "keywords": ["patient", "records", "found", "qatar"]
        },
        {
            "query": "What are the benefits of premium tier insurance?",
            "keywords": ["premium", "benefit", "coverage", "waiting"]
        },
        {
            "query": "Emergency services coverage",
            "keywords": ["emergency", "24/7", "available"]
        },
        {
            "query": "Cost for internal medicine consultation",
            "keywords": ["cost", "internal medicine", "tier", "estimate"]
        },
        {
            "query": "Find male patients with diabetes",
            "keywords": ["patient", "found", "records"]
        },
        {
            "query": "Book appointment this week",
            "keywords": ["appointment", "available", "found"]
        }
    ]
    
    print_header("Running Test Queries")
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{YELLOW}Test {i}/{len(test_cases)}{RESET}")
        if test_query(test_case["query"], test_case["keywords"]):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print_header("Test Summary")
    print(f"Total tests: {len(test_cases)}")
    print_success(f"Passed: {passed}")
    if failed > 0:
        print_error(f"Failed: {failed}")
    else:
        print_info("All tests passed!")
    
    # Performance test
    print_header("Performance Test")
    import time
    
    start_time = time.time()
    for _ in range(10):
        requests.post(f"{API_URL}/query", json={"query": "test"})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    print_info(f"Average response time: {avg_time:.3f} seconds")
    
    if avg_time < 1:
        print_success("Performance: Excellent")
    elif avg_time < 2:
        print_info("Performance: Good")
    else:
        print_error("Performance: Needs optimization")

if __name__ == "__main__":
    main()
