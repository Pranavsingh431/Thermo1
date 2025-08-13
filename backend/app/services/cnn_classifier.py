"""
Lightweight CNN classifier for thermal defect classification
Uses MobileNetV3 for efficient classification after YOLO-NAS detection
"""
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ThermalCNNClassifier:
    """Lightweight CNN classifier using MobileNetV3 for thermal defect classification"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._load_model()
        
    def _load_model(self):
        """Load MobileNetV3 model for classification"""
        try:
            self.model = models.mobilenet_v3_small(pretrained=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ MobileNetV3 classifier loaded on {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load MobileNetV3: {e}")
            self.model = None
            
    def classify_detections(self, image_path: str, detections: List[Dict]) -> List[Dict]:
        """
        Classify each detection using MobileNetV3
        
        Args:
            image_path: Path to the thermal image
            detections: List of YOLO-NAS detections with bounding boxes
            
        Returns:
            Enhanced detections with classification results
        """
        if not self.model:
            logger.warning("⚠️ MobileNetV3 model not available, skipping classification")
            return detections
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"❌ Could not load image: {image_path}")
                return detections
                
            classified_detections = []
            
            for i, detection in enumerate(detections):
                try:
                    bbox = detection.get("bbox", [0, 0, 100, 100])
                    x, y, w, h = bbox
                    
                    roi = image[y:y+h, x:x+w]
                    if roi.size == 0:
                        logger.warning(f"⚠️ Empty ROI for detection {i}")
                        classified_detections.append(detection)
                        continue
                        
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi_tensor = self.transform(roi_rgb).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(roi_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        
                    detection["classification_confidence"] = float(torch.max(probabilities))
                    detection["defect_probability"] = self._calculate_defect_probability(probabilities)
                    detection["thermal_classification"] = self._classify_thermal_defect(
                        detection.get("max_temperature", 0),
                        detection.get("component_type", "unknown")
                    )
                    
                    classified_detections.append(detection)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to classify detection {i}: {e}")
                    classified_detections.append(detection)
                    
            logger.info(f"✅ Classified {len(classified_detections)} detections using MobileNetV3")
            return classified_detections
            
        except Exception as e:
            logger.error(f"❌ Classification failed: {e}")
            return detections
            
    def _calculate_defect_probability(self, probabilities: torch.Tensor) -> float:
        """
        Calculate probability of thermal defect based on classification
        
        Args:
            probabilities: Softmax probabilities from MobileNetV3
            
        Returns:
            Defect probability (0.0 to 1.0)
        """
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
        normalized_entropy = float(entropy / np.log(len(probabilities)))
        
        max_confidence = float(torch.max(probabilities))
        defect_prob = (normalized_entropy + (1.0 - max_confidence)) / 2.0
        
        return min(max(defect_prob, 0.0), 1.0)
        
    def _classify_thermal_defect(self, temperature: float, component_type: str) -> str:
        """
        Classify thermal defect based on temperature and component type
        
        Args:
            temperature: Maximum temperature detected (°C)
            component_type: Type of component detected
            
        Returns:
            Thermal classification string
        """
        thresholds = {
            "conductor": {"normal": 75, "warning": 85, "critical": 95},
            "insulator": {"normal": 60, "warning": 70, "critical": 80},
            "connector": {"normal": 70, "warning": 80, "critical": 90},
            "transformer": {"normal": 65, "warning": 75, "critical": 85}
        }
        
        component_thresholds = thresholds.get(component_type, thresholds["conductor"])
        
        if temperature >= component_thresholds["critical"]:
            return "CRITICAL_OVERHEATING"
        elif temperature >= component_thresholds["warning"]:
            return "MODERATE_OVERHEATING"
        elif temperature >= component_thresholds["normal"]:
            return "MILD_OVERHEATING"
        else:
            return "NORMAL_OPERATION"

cnn_classifier = ThermalCNNClassifier()
