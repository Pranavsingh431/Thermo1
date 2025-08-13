#!/usr/bin/env python3
"""Test OpenRouter LLM integration"""

import sys
import os
sys.path.append('backend')

from dotenv import load_dotenv
load_dotenv('backend/.env')

from backend.app.utils.llm_openrouter import OpenRouterClient

def test_openrouter_llm():
    print("🤖 Testing OpenRouter LLM Integration")
    print("=" * 50)
    
    try:
        client = OpenRouterClient()
        
        test_data = {
            'max_temperature': 85.5,
            'components_detected': 3,
            'risk_level': 'high',
            'weather_context': {
                'mumbai_temp': 32.0,
                'humidity': 75,
                'condition': 'partly cloudy'
            }
        }
        
        print("Generating thermal inspection report...")
        result = client.generate_analysis(test_data)
        
        if result and 'report' in result:
            print("✅ OpenRouter LLM Success!")
            print(f"Model used: {result.get('model_used', 'unknown')}")
            print(f"Report length: {len(result['report'])} characters")
            print("\nSample report excerpt:")
            print("-" * 30)
            print(result['report'][:300] + "..." if len(result['report']) > 300 else result['report'])
            return True
        else:
            print("❌ OpenRouter LLM Failed - No report generated")
            return False
            
    except Exception as e:
        print(f"❌ OpenRouter LLM Error: {e}")
        return False

if __name__ == "__main__":
    success = test_openrouter_llm()
    sys.exit(0 if success else 1)
