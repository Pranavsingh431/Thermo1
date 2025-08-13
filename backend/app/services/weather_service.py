"""
Weather API service for Mumbai ambient temperature comparison
Integrates with OpenWeatherMap API for real-time weather data
"""
import requests
import logging
from typing import Optional, Dict
from app.config import settings

logger = logging.getLogger(__name__)

class WeatherService:
    """Service for fetching Mumbai weather data and temperature comparison"""
    
    def __init__(self):
        self.api_key = settings.WEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.mumbai_coords = {"lat": 19.1262, "lon": 72.8897}
        
    def get_mumbai_weather(self) -> Optional[Dict]:
        """Get current weather data for Mumbai, India"""
        try:
            lat, lon = self.mumbai_coords["lat"], self.mumbai_coords["lon"]
            url = f"{self.base_url}?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description'],
                'location': 'Mumbai, India',
                'timestamp': data['dt']
            }
            
            logger.info(f"✅ Mumbai weather: {weather_data['temperature']}°C, {weather_data['description']}")
            return weather_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to fetch Mumbai weather (network): {e}")
            return None
        except KeyError as e:
            logger.error(f"❌ Failed to parse weather data: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching weather: {e}")
            return None
            
    def is_fault(self, temp_detected: float, ambient_temp: float, threshold: float = 3.0) -> bool:
        """
        Determine if temperature difference indicates a thermal fault
        
        Args:
            temp_detected: Temperature detected in thermal image (°C)
            ambient_temp: Current ambient temperature (°C)
            threshold: Temperature difference threshold for fault detection (°C)
            
        Returns:
            True if temperature difference indicates a fault
        """
        temp_diff = temp_detected - ambient_temp
        is_fault_detected = temp_diff >= threshold
        
        logger.debug(f"Fault check: {temp_detected}°C - {ambient_temp}°C = {temp_diff}°C (threshold: {threshold}°C) -> {'FAULT' if is_fault_detected else 'NORMAL'}")
        
        return is_fault_detected
        
    def get_ambient_temperature_with_fallback(self) -> float:
        """Get Mumbai ambient temperature with fallback to default"""
        weather_data = self.get_mumbai_weather()
        
        if weather_data and 'temperature' in weather_data:
            return weather_data['temperature']
        else:
            logger.warning(f"⚠️ Using fallback ambient temperature: {settings.AMBIENT_TEMPERATURE_DEFAULT}°C")
            return settings.AMBIENT_TEMPERATURE_DEFAULT

weather_service = WeatherService()
