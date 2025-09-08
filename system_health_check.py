#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Precision Healthcare AI - System Health Check
Validates all servers and connections before deployment
"""
import httpx
import asyncio
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class HealthChecker:
    def __init__(self):
        self.servers = {
            "RAG Orchestrator": "http://localhost:8001",
            "Insurance Server": "http://localhost:8002",
            "Hospital Server": "http://localhost:8003"
        }
        self.test_queries = [
            {"query": "What is premium insurance coverage?"},
            {"query": "Book cardiology appointment"},
            {"query": "Find patients in Qatar"},
            {"query": "Emergency services"},
            {"query": "Pre-authorization requirements"}
        ]
        self.issues = []
        self.warnings = []
        
    async def check_server_health(self, name: str, url: str) -> Tuple[bool, str]:
        """Check if a server is running and healthy"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/health", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    return True, f"✓ Healthy - {data.get('patients_loaded', 0)} records"
                else:
                    return False, f"✗ Unhealthy - Status: {response.status_code}"
        except Exception as e:
            return False, f"✗ Offline - {str(e)[:50]}"
    
    async def check_inter_server_communication(self) -> Dict[str, bool]:
        """Test communication between servers"""
        results = {}
        
        # Test RAG -> Insurance
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.servers['RAG Orchestrator']}/query",
                    json={"query": "insurance coverage premium"},
                    timeout=5.0
                )
                results["RAG -> Insurance"] = response.status_code == 200
        except:
            results["RAG -> Insurance"] = False
            
        # Test RAG -> Hospital
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.servers['RAG Orchestrator']}/query",
                    json={"query": "book appointment cardiology"},
                    timeout=5.0
                )
                results["RAG -> Hospital"] = response.status_code == 200
        except:
            results["RAG -> Hospital"] = False
            
        return results
    
    async def test_queries(self) -> Dict[str, bool]:
        """Test sample queries"""
        results = {}
        
        async with httpx.AsyncClient() as client:
            for query in self.test_queries:
                try:
                    response = await client.post(
                        f"{self.servers['RAG Orchestrator']}/query",
                        json=query,
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        # Check if response has meaningful content
                        has_answer = bool(data.get('answer'))
                        has_context = bool(data.get('context'))
                        results[query['query']] = has_answer or has_context
                    else:
                        results[query['query']] = False
                except:
                    results[query['query']] = False
                    
        return results
    
    async def check_data_files(self) -> Dict[str, bool]:
        """Check if all required data files exist"""
        import os
        
        required_files = [
            "data/patient_augmented_synthetic.csv",
            "data/insurance_policies_gcc.csv",
            "data/cross_reference_table.csv",
            "data/insurance_coverage_matrix.csv",
            "data/specialty_procedures.csv",
            "data/preauthorization_rules.csv",
            "data/regulatory_authorities.csv"
        ]
        
        results = {}
        for file in required_files:
            results[file] = os.path.exists(file)
            if not results[file]:
                self.issues.append(f"Missing data file: {file}")
                
        return results
    
    async def run_complete_check(self):
        """Run all health checks"""
        print(f"{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}Precision Healthcare AI - System Health Check{RESET}")
        print(f"{BLUE}{'='*60}{RESET}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 1. Check servers
        print(f"{YELLOW}1. Server Status:{RESET}")
        all_healthy = True
        for name, url in self.servers.items():
            healthy, message = await self.check_server_health(name, url)
            if healthy:
                print(f"   {GREEN}{name}: {message}{RESET}")
            else:
                print(f"   {RED}{name}: {message}{RESET}")
                all_healthy = False
                self.issues.append(f"{name} is not healthy")
        
        # 2. Check inter-server communication
        print(f"\n{YELLOW}2. Inter-Server Communication:{RESET}")
        comm_results = await self.check_inter_server_communication()
        for connection, status in comm_results.items():
            if status:
                print(f"   {GREEN}✓ {connection}{RESET}")
            else:
                print(f"   {RED}✗ {connection}{RESET}")
                self.issues.append(f"Communication failure: {connection}")
        
        # 3. Test queries
        print(f"\n{YELLOW}3. Query Tests:{RESET}")
        query_results = await self.test_queries()
        for query, success in query_results.items():
            if success:
                print(f"   {GREEN}✓ {query[:40]}...{RESET}")
            else:
                print(f"   {RED}✗ {query[:40]}...{RESET}")
                self.warnings.append(f"Query failed: {query}")
        
        # 4. Check data files
        print(f"\n{YELLOW}4. Data Files:{RESET}")
        file_results = await self.check_data_files()
        files_ok = all(file_results.values())
        if files_ok:
            print(f"   {GREEN}✓ All {len(file_results)} data files present{RESET}")
        else:
            missing = [f for f, exists in file_results.items() if not exists]
            print(f"   {RED}✗ Missing {len(missing)} files{RESET}")
            for f in missing[:3]:
                print(f"      - {f}")
        
        # 5. Summary
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}Summary:{RESET}")
        print(f"{BLUE}{'='*60}{RESET}")
        
        if not self.issues and not self.warnings:
            print(f"{GREEN}✓ System is READY for deployment!{RESET}")
            return True
        else:
            if self.issues:
                print(f"{RED}Critical Issues ({len(self.issues)}):{RESET}")
                for issue in self.issues[:5]:
                    print(f"   - {issue}")
            
            if self.warnings:
                print(f"{YELLOW}Warnings ({len(self.warnings)}):{RESET}")
                for warning in self.warnings[:5]:
                    print(f"   - {warning}")
            
            print(f"\n{RED}✗ System needs fixes before deployment{RESET}")
            return False

async def main():
    checker = HealthChecker()
    ready = await checker.run_complete_check()
    
    if ready:
        print(f"\n{GREEN}Next steps:{RESET}")
        print("1. Run: python create_deployment_package.py")
        print("2. Commit to GitHub")
        print("3. Deploy to cloud")
    else:
        print(f"\n{YELLOW}Fix the issues above, then run this check again{RESET}")
    
    sys.exit(0 if ready else 1)

if __name__ == "__main__":
    asyncio.run(main())
