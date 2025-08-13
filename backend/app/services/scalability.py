import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import redis
import json
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models.thermal_scan import ThermalScan
from app.models.ai_analysis import AIAnalysis

logger = logging.getLogger(__name__)

class ScalabilityService:
    def __init__(self):
        self.redis_client = None
        self.connection_pool = None
        self.rate_limits = {
            'upload': 100,  # per minute per user
            'analysis': 50,  # per minute per user
            'reports': 20   # per minute per user
        }
        
    async def initialize_redis(self):
        """Initialize Redis connection for caching and rate limiting"""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def check_rate_limit(self, user_id: str, action: str) -> bool:
        """Check if user has exceeded rate limits"""
        if not self.redis_client:
            return True  # Allow if Redis unavailable
            
        try:
            key = f"rate_limit:{user_id}:{action}"
            current_count = await self.redis_client.get(key)
            
            if current_count is None:
                await self.redis_client.setex(key, 60, 1)
                return True
            
            if int(current_count) >= self.rate_limits.get(action, 100):
                return False
                
            await self.redis_client.incr(key)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error
    
    async def cache_analysis_result(self, image_id: str, result: Dict[str, Any], ttl: int = 3600):
        """Cache AI analysis results"""
        if not self.redis_client:
            return
            
        try:
            key = f"analysis_cache:{image_id}"
            await self.redis_client.setex(key, ttl, json.dumps(result))
            logger.info(f"Cached analysis result for image {image_id}")
        except Exception as e:
            logger.error(f"Failed to cache analysis result: {e}")
    
    async def get_cached_analysis(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis result"""
        if not self.redis_client:
            return None
            
        try:
            key = f"analysis_cache:{image_id}"
            cached_result = await self.redis_client.get(key)
            
            if cached_result:
                return json.loads(cached_result)
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve cached analysis: {e}")
            return None
    
    async def queue_batch_processing(self, image_ids: List[str], priority: str = "normal"):
        """Queue multiple images for batch processing"""
        if not self.redis_client:
            return False
            
        try:
            queue_name = f"batch_queue:{priority}"
            batch_data = {
                'image_ids': image_ids,
                'queued_at': datetime.utcnow().isoformat(),
                'priority': priority
            }
            
            await self.redis_client.lpush(queue_name, json.dumps(batch_data))
            logger.info(f"Queued {len(image_ids)} images for batch processing")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue batch processing: {e}")
            return False
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            with next(get_db()) as db:
                total_scans = db.query(func.count(ThermalScan.id)).scalar()
                recent_scans = db.query(func.count(ThermalScan.id)).filter(
                    ThermalScan.created_at >= datetime.utcnow() - timedelta(hours=24)
                ).scalar()
                
                processing_scans = db.query(func.count(ThermalScan.id)).filter(
                    ThermalScan.status == 'processing'
                ).scalar()
                
                redis_info = {}
                if self.redis_client:
                    redis_info = await self.redis_client.info()
                
                return {
                    'database': {
                        'total_scans': total_scans,
                        'recent_scans_24h': recent_scans,
                        'processing_scans': processing_scans
                    },
                    'redis': {
                        'connected': self.redis_client is not None,
                        'memory_usage': redis_info.get('used_memory_human', 'N/A'),
                        'connected_clients': redis_info.get('connected_clients', 0)
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {'error': str(e)}
    
    async def optimize_database_queries(self, db: Session):
        """Optimize database performance for high load"""
        try:
            db.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thermal_scan_created_at ON thermal_scans(created_at)")
            db.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_thermal_scan_status ON thermal_scans(status)")
            db.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ai_analysis_confidence ON ai_analyses(confidence_score)")
            
            db.execute("ANALYZE thermal_scans")
            db.execute("ANALYZE ai_analyses")
            
            db.commit()
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            db.rollback()
    
    async def handle_concurrent_uploads(self, upload_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle multiple concurrent upload requests efficiently"""
        try:
            semaphore = asyncio.Semaphore(10)  # Limit concurrent processing
            
            async def process_upload(upload_data):
                async with semaphore:
                    await asyncio.sleep(0.1)
                    return {
                        'upload_id': upload_data.get('id'),
                        'status': 'queued',
                        'timestamp': datetime.utcnow().isoformat()
                    }
            
            tasks = [process_upload(upload) for upload in upload_requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return [r for r in results if not isinstance(r, Exception)]
            
        except Exception as e:
            logger.error(f"Concurrent upload handling failed: {e}")
            return []
    
    async def auto_scale_workers(self, current_load: int, target_load: int = 80):
        """Auto-scale worker processes based on current load"""
        try:
            if current_load > target_load:
                logger.info(f"High load detected ({current_load}%), considering scale up")
                return {'action': 'scale_up', 'current_load': current_load}
            elif current_load < target_load * 0.5:
                logger.info(f"Low load detected ({current_load}%), considering scale down")
                return {'action': 'scale_down', 'current_load': current_load}
            else:
                return {'action': 'maintain', 'current_load': current_load}
                
        except Exception as e:
            logger.error(f"Auto-scaling check failed: {e}")
            return {'action': 'error', 'error': str(e)}

scalability_service = ScalabilityService()
