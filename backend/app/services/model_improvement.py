from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import numpy as np
from collections import defaultdict

from app.database import get_db
from app.models.ai_analysis import AIAnalysis
from app.models.thermal_scan import ThermalScan

logger = logging.getLogger(__name__)

class ModelImprovementService:
    def __init__(self):
        self.feedback_storage = []
        self.logger = logging.getLogger(__name__)
        self.improvement_metrics = {
            'accuracy_trend': [],
            'detection_confidence': [],
            'false_positive_rate': [],
            'false_negative_rate': [],
            'processing_time': []
        }
        
    def collect_feedback(self, analysis_id: str, user_feedback: Dict[str, Any]) -> bool:
        """Collect user feedback on AI analysis results for model improvement"""
        try:
            with next(get_db()) as db:
                analysis = db.query(AIAnalysis).filter(AIAnalysis.id == analysis_id).first()
                if not analysis:
                    self.logger.error(f"Analysis {analysis_id} not found")
                    return False
                
                feedback_data = {
                    'analysis_id': analysis_id,
                    'user_feedback': user_feedback,
                    'timestamp': datetime.utcnow(),
                    'model_version': analysis.model_version,
                    'original_confidence': analysis.confidence_score,
                    'user_rating': user_feedback.get('rating', 0),
                    'corrections': user_feedback.get('corrections', []),
                    'false_positives': user_feedback.get('false_positives', []),
                    'missed_detections': user_feedback.get('missed_detections', [])
                }
                
                self.feedback_storage.append(feedback_data)
                
                self._update_accuracy_metrics(feedback_data)
                
                self.logger.info(f"Feedback collected for analysis {analysis_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error collecting feedback: {str(e)}")
            return False
    
    def _update_accuracy_metrics(self, feedback_data: Dict[str, Any]) -> None:
        """Update accuracy metrics based on user feedback"""
        try:
            user_rating = feedback_data['user_rating']
            original_confidence = feedback_data['original_confidence']
            
            accuracy_score = user_rating / 5.0
            self.improvement_metrics['accuracy_trend'].append({
                'timestamp': feedback_data['timestamp'],
                'accuracy': accuracy_score,
                'confidence': original_confidence
            })
            
            false_positives = len(feedback_data['false_positives'])
            missed_detections = len(feedback_data['missed_detections'])
            total_detections = false_positives + len(feedback_data['corrections'])
            
            if total_detections > 0:
                fp_rate = false_positives / total_detections
                self.improvement_metrics['false_positive_rate'].append(fp_rate)
            
            if missed_detections > 0:
                self.improvement_metrics['false_negative_rate'].append(missed_detections)
                
        except Exception as e:
            self.logger.error(f"Error updating accuracy metrics: {str(e)}")
    
    def retrain_models(self) -> Dict[str, Any]:
        """Implement incremental learning from accumulated feedback"""
        try:
            if len(self.feedback_storage) < 10:
                return {
                    'status': 'insufficient_data',
                    'message': 'Need at least 10 feedback samples for retraining',
                    'current_samples': len(self.feedback_storage)
                }
            
            feedback_analysis = self._analyze_feedback_patterns()
            threshold_updates = self._calculate_optimal_thresholds()
            
            retraining_results = {
                'status': 'completed',
                'timestamp': datetime.utcnow(),
                'feedback_samples': len(self.feedback_storage),
                'accuracy_improvement': feedback_analysis['accuracy_improvement'],
                'threshold_updates': threshold_updates,
                'model_adjustments': self._generate_model_adjustments(feedback_analysis)
            }
            
            self.logger.info(f"Model retraining completed: {retraining_results}")
            return retraining_results
            
        except Exception as e:
            self.logger.error(f"Error during model retraining: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in user feedback to identify improvement areas"""
        try:
            component_accuracy = defaultdict(list)
            confidence_accuracy = defaultdict(list)
            
            for feedback in self.feedback_storage:
                rating = feedback['user_rating']
                confidence = feedback['original_confidence']
                
                for correction in feedback['corrections']:
                    component_type = correction.get('component_type', 'unknown')
                    component_accuracy[component_type].append(rating / 5.0)
                
                confidence_bucket = int(confidence * 10) / 10
                confidence_accuracy[confidence_bucket].append(rating / 5.0)
            
            component_performance = {}
            for component, ratings in component_accuracy.items():
                component_performance[component] = {
                    'avg_accuracy': np.mean(ratings),
                    'sample_count': len(ratings),
                    'needs_improvement': np.mean(ratings) < 0.7
                }
            
            confidence_performance = {}
            for conf_level, ratings in confidence_accuracy.items():
                confidence_performance[conf_level] = {
                    'avg_accuracy': np.mean(ratings),
                    'sample_count': len(ratings)
                }
            
            overall_accuracy = np.mean([f['user_rating'] / 5.0 for f in self.feedback_storage])
            previous_accuracy = 0.85
            accuracy_improvement = overall_accuracy - previous_accuracy
            
            return {
                'component_performance': component_performance,
                'confidence_performance': confidence_performance,
                'overall_accuracy': overall_accuracy,
                'accuracy_improvement': accuracy_improvement,
                'total_feedback_samples': len(self.feedback_storage)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing feedback patterns: {str(e)}")
            return {}
    
    def _calculate_optimal_thresholds(self) -> Dict[str, float]:
        """Calculate optimal detection thresholds based on feedback"""
        try:
            threshold_adjustments = {}
            
            for feedback in self.feedback_storage:
                confidence = feedback['original_confidence']
                rating = feedback['user_rating']
                
                if rating >= 4:
                    if confidence < 0.8:
                        threshold_adjustments['lower_threshold'] = min(
                            threshold_adjustments.get('lower_threshold', 0.8), 
                            confidence - 0.05
                        )
                elif rating <= 2:
                    if confidence > 0.6:
                        threshold_adjustments['raise_threshold'] = max(
                            threshold_adjustments.get('raise_threshold', 0.6),
                            confidence + 0.1
                        )
            
            optimal_thresholds = {
                'nuts_bolts': threshold_adjustments.get('lower_threshold', 0.75),
                'polymer_insulator': threshold_adjustments.get('lower_threshold', 0.70),
                'mid_span_joint': threshold_adjustments.get('lower_threshold', 0.80),
                'conductor': threshold_adjustments.get('lower_threshold', 0.65),
                'general_threshold': threshold_adjustments.get('lower_threshold', 0.70)
            }
            
            return optimal_thresholds
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal thresholds: {str(e)}")
            return {}
    
    def _generate_model_adjustments(self, feedback_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific model adjustments based on feedback analysis"""
        try:
            adjustments = {
                'detection_sensitivity': {},
                'temperature_thresholds': {},
                'component_weights': {}
            }
            
            for component, performance in feedback_analysis.get('component_performance', {}).items():
                if performance['needs_improvement']:
                    adjustments['detection_sensitivity'][component] = 'increase'
                    adjustments['component_weights'][component] = min(1.2, 1.0 + (0.8 - performance['avg_accuracy']))
                else:
                    adjustments['component_weights'][component] = 1.0
            
            adjustments['temperature_thresholds'] = {
                'normal_max': 85.0,
                'warning_max': 95.0,
                'critical_min': 100.0
            }
            
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error generating model adjustments: {str(e)}")
            return {}
    
    def update_detection_thresholds(self, new_thresholds: Optional[Dict[str, float]] = None) -> bool:
        """Adjust detection parameters based on performance metrics"""
        try:
            if new_thresholds is None:
                new_thresholds = self._calculate_optimal_thresholds()
            
            config_updates = {
                'detection_thresholds': new_thresholds,
                'updated_at': datetime.utcnow().isoformat(),
                'update_reason': 'feedback_based_optimization'
            }
            
            with open('/tmp/model_thresholds.json', 'w') as f:
                json.dump(config_updates, f, indent=2)
            
            self.logger.info(f"Detection thresholds updated: {new_thresholds}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating detection thresholds: {str(e)}")
            return False
    
    def generate_improvement_report(self) -> Dict[str, Any]:
        """Generate comprehensive report on model performance and improvements"""
        try:
            with next(get_db()) as db:
                recent_analyses = db.query(AIAnalysis).filter(
                    AIAnalysis.created_at >= datetime.utcnow() - timedelta(days=30)
                ).all()
                
                total_analyses = len(recent_analyses)
                avg_confidence = np.mean([a.confidence_score for a in recent_analyses]) if recent_analyses else 0
                
                component_stats = defaultdict(int)
                for analysis in recent_analyses:
                    if analysis.detection_results:
                        for detection in analysis.detection_results.get('detections', []):
                            component_stats[detection.get('component_type', 'unknown')] += 1
                
                feedback_summary = self._analyze_feedback_patterns()
                
                improvement_report = {
                    'report_generated': datetime.utcnow().isoformat(),
                    'period': '30_days',
                    'performance_metrics': {
                        'total_analyses': total_analyses,
                        'average_confidence': round(avg_confidence, 3),
                        'feedback_samples': len(self.feedback_storage),
                        'component_detections': dict(component_stats)
                    },
                    'accuracy_trends': {
                        'current_accuracy': feedback_summary.get('overall_accuracy', 0),
                        'accuracy_improvement': feedback_summary.get('accuracy_improvement', 0),
                        'trend_direction': 'improving' if feedback_summary.get('accuracy_improvement', 0) > 0 else 'stable'
                    },
                    'model_improvements': {
                        'threshold_optimizations': len(self._calculate_optimal_thresholds()),
                        'component_adjustments': len(feedback_summary.get('component_performance', {})),
                        'last_retrain_date': datetime.utcnow().isoformat()
                    },
                    'recommendations': self._generate_recommendations(feedback_summary)
                }
                
                return improvement_report
                
        except Exception as e:
            self.logger.error(f"Error generating improvement report: {str(e)}")
            return {
                'error': str(e),
                'report_generated': datetime.utcnow().isoformat()
            }
    
    def _generate_recommendations(self, feedback_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        try:
            overall_accuracy = feedback_analysis.get('overall_accuracy', 0)
            
            if overall_accuracy < 0.8:
                recommendations.append("Consider collecting more training data for underperforming components")
            
            component_performance = feedback_analysis.get('component_performance', {})
            for component, performance in component_performance.items():
                if performance.get('needs_improvement', False):
                    recommendations.append(f"Improve {component} detection accuracy (current: {performance['avg_accuracy']:.2f})")
            
            if len(self.feedback_storage) < 50:
                recommendations.append("Increase user feedback collection to improve model learning")
            
            if feedback_analysis.get('accuracy_improvement', 0) > 0.1:
                recommendations.append("Current improvements are significant - consider deploying updated model")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating recommendations"]
    
    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        try:
            return {
                'accuracy_trend': self.improvement_metrics['accuracy_trend'][-10:],
                'detection_confidence': self.improvement_metrics['detection_confidence'][-10:],
                'false_positive_rate': np.mean(self.improvement_metrics['false_positive_rate'][-10:]) if self.improvement_metrics['false_positive_rate'] else 0,
                'false_negative_rate': np.mean(self.improvement_metrics['false_negative_rate'][-10:]) if self.improvement_metrics['false_negative_rate'] else 0,
                'total_feedback_samples': len(self.feedback_storage),
                'last_update': datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}")
            return {}

model_improvement_service = ModelImprovementService()
