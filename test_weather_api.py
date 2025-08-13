#!/usr/bin/env python3
"""Test weather API integration"""

import sys
import os
sys.path.append('backend')

from dotenv import load_dotenv
load_dotenv('backend/.env')

from backend.app.services.weather_service import weather_service

def test_weather_api():
    print("🌡️ Testing Mumbai Weather API Integration")
    print("=" * 50)
    
    weather_data = weather_service.get_mumbai_weather()
    
    if weather_data:
        print("✅ Weather API Success!")
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Wind Speed: {weather_data['wind_speed']} m/s")
        print(f"Condition: {weather_data['description']}")
        print(f"Location: {weather_data['location']}")
        
        test_temp = 45.0
        is_fault = weather_service.is_fault(test_temp, weather_data['temperature'])
        print(f"\nFault Detection Test:")
        print(f"Component temp: {test_temp}°C vs Ambient: {weather_data['temperature']}°C")
        print(f"Is Fault: {'YES' if is_fault else 'NO'}")
        
        return True
    else:
        print("❌ Weather API Failed!")
        return False

if __name__ == "__main__":
    success = test_weather_api()
    sys.exit(0 if success else 1)
