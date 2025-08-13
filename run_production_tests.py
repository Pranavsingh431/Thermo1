#!/usr/bin/env python3
"""
Production readiness test runner
"""
import os
import sys
import subprocess
import asyncio
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run shell command and return result"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=300
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def main():
    """Run all production tests"""
    print("🚀 THERMAL INSPECTION SYSTEM - PRODUCTION TESTS")
    print("=" * 60)
    
    os.chdir(Path(__file__).parent)
    
    tests = [
        ("Install AI Dependencies", "pip install -r backend/requirements-ai.txt"),
        ("Install Backend Dependencies", "pip install -r backend/requirements.txt"),
        ("Install Frontend Dependencies", "cd frontend && npm install"),
        ("AI Pipeline Test", "cd backend && python test_ai_pipeline.py"),
        ("Frontend Build Test", "cd frontend && npm run build"),
        ("Production Readiness Check", "python verify_production_readiness.py")
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, command in tests:
        print(f"\n🔍 {test_name}...")
        success, stdout, stderr = run_command(command)
        
        if success:
            print(f"  ✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"  ❌ {test_name} FAILED")
            if stderr:
                print(f"     Error: {stderr[:200]}...")
    
    print("\n" + "=" * 60)
    print("PRODUCTION TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total} ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - READY FOR PRODUCTION!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED - NEEDS ATTENTION")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
