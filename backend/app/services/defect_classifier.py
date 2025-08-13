"""
IEEE-compliant defect classification for thermal inspection
Implements Tata Power's thermal analysis standards
"""
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DefectClassification:
    """IEEE-compliant defect classification result"""
    component_id: str
    component_type: str
    thermal_severity: str
    risk_level: str
    measured_temperature: float
    ambient_temperature: float
    temperature_rise: float
    threshold_exceeded: bool
    confidence_score: float = 0.0
    defect_description: str = ""
    recommended_action: str = ""
    weather_context: Optional[Dict] = None

class TataPowerDefectClassifier:
    """IEEE C57.91 compliant thermal defect classifier for Tata Power"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temperature_thresholds = {
            "conductor": {"normal": 75, "warning": 85, "critical": 95},
            "insulator": {"normal": 60, "warning": 70, "critical": 80},
            "connector": {"normal": 70, "warning": 80, "critical": 90},
            "transformer": {"normal": 65, "warning": 75, "critical": 85}
        }
        
    def classify_thermal_defects(self, detections: List[Dict], ambient_temp: float = 25.0) -> List[DefectClassification]:
        """Classify thermal defects according to IEEE standards with weather comparison"""
        classifications = []
        
        try:
            from app.services.weather_service import weather_service
            weather_data = weather_service.get_mumbai_weather()
            
            if weather_data:
                mumbai_ambient = weather_data['temperature']
                self.logger.info(f"🌡️ Mumbai ambient temperature: {mumbai_ambient}°C")
                if ambient_temp == 25.0:  # Default value
                    ambient_temp = mumbai_ambient
                    self.logger.info(f"✅ Using Mumbai ambient temperature: {ambient_temp}°C")
            
            for detection in detections:
                classification = self._classify_single_detection(detection, ambient_temp)
                if classification:
                    if weather_data:
                        classification.weather_context = {
                            'mumbai_temp': weather_data['temperature'],
                            'humidity': weather_data['humidity'],
                            'wind_speed': weather_data['wind_speed'],
                            'condition': weather_data['description']
                        }
                    classifications.append(classification)
                    
            self.logger.info(f"✅ Classified {len(classifications)} components using IEEE standards with weather data")
            return classifications
            
        except Exception as e:
            self.logger.error(f"❌ Classification failed: {e}")
            return []
    
    def _classify_single_detection(self, detection: Dict, ambient_temp: float) -> Optional[DefectClassification]:
        """Classify a single thermal detection"""
        try:
            component_type = detection.get('component_type', 'unknown')
            temp = detection.get('temperature', 0.0)
            component_id = detection.get('component_id', f"COMP_{len(detection)}")
            
            thresholds = self.temperature_thresholds.get(component_type, self.temperature_thresholds["conductor"])
            
            temperature_rise = temp - ambient_temp
            risk_level = self._determine_risk_level(temp, thresholds)
            thermal_severity = self._determine_thermal_severity(temp, thresholds)
            threshold_exceeded = temp > thresholds["normal"]
            
            return DefectClassification(
                component_id=component_id,
                component_type=component_type,
                thermal_severity=thermal_severity,
                risk_level=risk_level.value,
                measured_temperature=temp,
                ambient_temperature=ambient_temp,
                temperature_rise=temperature_rise,
                threshold_exceeded=threshold_exceeded,
                confidence_score=detection.get('confidence', 0.8),
                defect_description=self._generate_defect_description(temp, component_type, thresholds),
                recommended_action=self._generate_recommended_action(risk_level, component_type)
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to classify detection: {e}")
            return None
    
    def _determine_risk_level(self, temp: float, thresholds: Dict) -> RiskLevel:
        """Determine risk level based on temperature"""
        if temp >= thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif temp >= thresholds["warning"]:
            return RiskLevel.HIGH
        elif temp >= thresholds["normal"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _determine_thermal_severity(self, temp: float, thresholds: Dict) -> str:
        """Determine thermal severity classification"""
        if temp >= thresholds["critical"]:
            return "CRITICAL_OVERHEATING"
        elif temp >= thresholds["warning"]:
            return "MODERATE_OVERHEATING"
        elif temp >= thresholds["normal"]:
            return "MILD_OVERHEATING"
        else:
            return "NORMAL_OPERATION"
    
    def _generate_defect_description(self, temp: float, component_type: str, thresholds: Dict) -> str:
        """Generate IEEE-compliant defect description"""
        if temp >= thresholds["critical"]:
            return f"Critical thermal anomaly detected in {component_type}. Temperature {temp}°C exceeds critical threshold {thresholds['critical']}°C."
        elif temp >= thresholds["warning"]:
            return f"Moderate thermal anomaly in {component_type}. Temperature {temp}°C above warning threshold {thresholds['warning']}°C."
        elif temp >= thresholds["normal"]:
            return f"Mild thermal elevation in {component_type}. Temperature {temp}°C above normal threshold {thresholds['normal']}°C."
        else:
            return f"{component_type} operating within normal thermal parameters at {temp}°C."
    
    def _generate_recommended_action(self, risk_level: RiskLevel, component_type: str) -> str:
        """Generate recommended maintenance actions"""
        actions = {
            RiskLevel.CRITICAL: f"IMMEDIATE ACTION REQUIRED: Shut down {component_type} and inspect for loose connections, corrosion, or overloading.",
            RiskLevel.HIGH: f"Schedule urgent maintenance for {component_type}. Inspect connections and load conditions within 24 hours.",
            RiskLevel.MEDIUM: f"Monitor {component_type} closely. Schedule routine maintenance within 1 week.",
            RiskLevel.LOW: f"{component_type} operating normally. Continue routine monitoring."
        }
        return actions.get(risk_level, "Monitor component condition.")

tata_power_defect_classifier = TataPowerDefectClassifier()
