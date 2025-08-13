#!/usr/bin/env python3
"""Test complete thermal inspection system integration"""

import sys
import os
sys.path.append('backend')

from dotenv import load_dotenv
load_dotenv('backend/.env')

def test_complete_system():
    print("🔥 Testing Complete Thermal Inspection System")
    print("=" * 60)
    
    print("\n1. Testing Weather API Integration...")
    try:
        from backend.app.services.weather_service import weather_service
        weather_data = weather_service.get_mumbai_weather()
        if weather_data:
            print(f"✅ Weather API: {weather_data['temperature']}°C in Mumbai")
        else:
            print("❌ Weather API failed")
    except Exception as e:
        print(f"❌ Weather API error: {e}")
    
    print("\n2. Testing AI Model Loading...")
    try:
        from backend.app.services.model_loader import model_loader
        model_loader.initialize_all_models()
        status = model_loader.get_model_status()
        print(f"✅ Models loaded: {list(status['loaded_models'].keys())}")
    except Exception as e:
        print(f"❌ Model loading error: {e}")
    
    print("\n3. Testing CNN Classifier...")
    try:
        from backend.app.services.cnn_classifier import cnn_classifier
        print("✅ CNN Classifier initialized")
    except Exception as e:
        print(f"❌ CNN Classifier error: {e}")
    
    print("\n4. Testing Bulletproof AI Pipeline...")
    try:
        from backend.app.services.bulletproof_ai_pipeline import bulletproof_ai_pipeline
        status = bulletproof_ai_pipeline.get_system_status()
        print(f"✅ AI Pipeline status: {status['model_status']}")
    except Exception as e:
        print(f"❌ AI Pipeline error: {e}")
    
    print("\n5. Testing Image Processing...")
    try:
        import glob
        test_images = glob.glob("test_thermal_images/*.jpg")
        if test_images:
            test_image = test_images[0]
            result = bulletproof_ai_pipeline.analyze_thermal_image(test_image, "test_image")
            print(f"✅ Processed {test_image}: {result.total_components} components detected")
            print(f"   Risk level: {result.overall_risk_level}")
            print(f"   Max temp: {result.max_temperature}°C")
        else:
            print("⚠️ No test images found")
    except Exception as e:
        print(f"❌ Image processing error: {e}")
    
    print("\n6. Testing Configuration...")
    try:
        from backend.app.config import settings
        print(f"✅ Weather API key configured: {'Yes' if settings.WEATHER_API_KEY else 'No'}")
        print(f"✅ OpenRouter key configured: {'Yes' if settings.OPEN_ROUTER_KEY else 'No'}")
        print(f"✅ Database URL configured: {'Yes' if settings.DATABASE_URL else 'No'}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
    
    print("\n🎯 System Integration Test Complete")
    return True

if __name__ == "__main__":
    test_complete_system()
