#!/usr/bin/env python3
"""
Test script to verify AI pipeline functionality
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.model_loader import model_loader
from app.services.bulletproof_ai_pipeline import BulletproofAIPipeline
from app.utils.llm_openrouter import OpenRouterClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_model_loading():
    """Test if AI models can be loaded successfully"""
    print("=" * 50)
    print("TESTING AI MODEL LOADING")
    print("=" * 50)
    
    try:
        status = model_loader.get_model_status()
        print(f"Model Status: {status}")
        
        if 'yolov8n' in status.get('loaded_models', []):
            print("✅ Ultralytics YOLO model is available")
        else:
            print("❌ Ultralytics YOLO model is NOT available")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

async def test_openrouter_integration():
    """Test OpenRouter API integration"""
    print("\n" + "=" * 50)
    print("TESTING OPENROUTER INTEGRATION")
    print("=" * 50)
    
    try:
        client = OpenRouterClient()
        
        test_prompt = "Analyze this thermal image data: Temperature readings show 85°C on conductor, 92°C on insulator. Provide engineering assessment."
        
        response = client.generate_analysis({"prompt": test_prompt})
        
        if response and isinstance(response, dict) and len(str(response)) > 50:
            print("✅ OpenRouter integration working")
            print(f"Sample response: {str(response)[:100]}...")
            return True
        else:
            print("❌ OpenRouter integration failed - empty or short response")
            return False
            
    except Exception as e:
        print(f"❌ OpenRouter test failed: {e}")
        return False

async def test_ai_pipeline():
    """Test the complete AI pipeline"""
    print("\n" + "=" * 50)
    print("TESTING COMPLETE AI PIPELINE")
    print("=" * 50)
    
    try:
        pipeline = BulletproofAIPipeline()
        
        import numpy as np
        from PIL import Image
        
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        dummy_image[100:150, 200:250] = [255, 0, 0]  # Red hot spot
        dummy_image[300:350, 400:450] = [255, 255, 0]  # Yellow warm spot
        
        pil_image = Image.fromarray(dummy_image)
        
        test_image_path = "test_thermal_sample.jpg"
        pil_image.save(test_image_path)
        
        result = pipeline.process_thermal_image(
            image_path=test_image_path,
            image_id="TEST-001",
            ambient_temp=35.0
        )
        
        if result and hasattr(result, 'detections'):
            print("✅ AI Pipeline working")
            print(f"Detections found: {len(result.detections)}")
            print(f"Overall status: {result.overall_risk_level}")
            return True
        else:
            print("❌ AI Pipeline failed - no detections returned")
            return False
            
    except Exception as e:
        print(f"❌ AI Pipeline test failed: {e}")
        return False

async def test_dependencies():
    """Test critical dependencies"""
    print("\n" + "=" * 50)
    print("TESTING DEPENDENCIES")
    print("=" * 50)
    
    dependencies = {
        'torch': 'torch',
        'torchvision': 'torchvision', 
        'ultralytics': 'ultralytics',
        'opencv': 'cv2',
        'PIL': 'PIL',
        'numpy': 'numpy'
    }
    
    results = {}
    
    for name, import_name in dependencies.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
            results[name] = True
        except ImportError as e:
            print(f"❌ {name}: NOT INSTALLED - {e}")
            results[name] = False
    
    return all(results.values())

async def main():
    """Run all tests"""
    print("THERMAL INSPECTION SYSTEM - AI PIPELINE VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies()),
        ("Model Loading", test_model_loading()),
        ("OpenRouter Integration", test_openrouter_integration()),
        ("AI Pipeline", test_ai_pipeline())
    ]
    
    results = []
    
    for test_name, test_coro in tests:
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - System is production ready!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED - System needs fixes before production")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
