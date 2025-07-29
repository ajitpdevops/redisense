"""
Redis service for data storage and vector operations
"""
import json
import logging
from typing import List, Dict, Optional, Any
import redis
import numpy as np
from datetime import datetime

from config.settings import Settings
from app.models.schemas import Device, EnergyReading, SemanticSearchResult

logger = logging.getLogger(__name__)

class RedisService:
    """Redis service for data operations and vector search"""

    def __init__(self, settings: Settings = None):
        """Initialize Redis connection"""
        self.settings = settings or Settings()
        self.redis_client = None
        self._connect()

    def _connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                username=self.settings.REDIS_USERNAME if self.settings.REDIS_USERNAME else None,
                password=self.settings.REDIS_PASSWORD if self.settings.REDIS_PASSWORD else None,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.settings.REDIS_HOST}:{self.settings.REDIS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def store_device(self, device: Device) -> bool:
        """Store device information"""
        try:
            device_key = f"device:{device.device_id}"
            device_data = device.model_dump(mode='json')  # Use JSON mode to properly serialize enums
            device_data['created_at'] = datetime.utcnow().isoformat()

            # Convert all values to strings for Redis storage
            redis_data = {}
            for key, value in device_data.items():
                if isinstance(value, datetime):
                    redis_data[key] = value.isoformat()
                elif isinstance(value, dict):
                    redis_data[key] = json.dumps(value)
                else:
                    redis_data[key] = str(value)

            self.redis_client.hset(device_key, mapping=redis_data)

            # Add to device index
            self.redis_client.sadd("devices:index", device.device_id)

            logger.info(f"Stored device {device.device_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing device {device.device_id}: {e}")
            return False

    def get_device(self, device_id: str) -> Optional[Device]:
        """Retrieve device by ID"""
        try:
            device_key = f"device:{device_id}"
            device_data = self.redis_client.hgetall(device_key)

            if not device_data:
                return None

            # Convert data back from Redis strings
            parsed_data = {}
            for key, value in device_data.items():
                if key == 'install_date' or key == 'created_at':
                    try:
                        parsed_data[key] = datetime.fromisoformat(value)
                    except:
                        parsed_data[key] = datetime.utcnow()
                elif key == 'metadata':
                    try:
                        parsed_data[key] = json.loads(value)
                    except:
                        parsed_data[key] = {}
                else:
                    parsed_data[key] = value

            # Convert back to Device model
            return Device(**parsed_data)
        except Exception as e:
            logger.error(f"Error retrieving device {device_id}: {e}")
            return None

    def get_all_devices(self) -> List[Device]:
        """Get all devices"""
        try:
            device_ids = self.redis_client.smembers("devices:index")
            devices = []

            for device_id in device_ids:
                device = self.get_device(device_id)
                if device:
                    devices.append(device)

            return devices
        except Exception as e:
            logger.error(f"Error retrieving all devices: {e}")
            return []

    def store_energy_reading(self, reading: EnergyReading) -> bool:
        """Store energy reading"""
        try:
            # Store in time series
            reading_key = f"energy:{reading.device_id}:{reading.timestamp.isoformat()}"
            reading_data = reading.model_dump(mode='json')  # Use JSON mode for proper serialization

            # Convert all values to strings for Redis storage
            redis_data = {}
            for key, value in reading_data.items():
                if isinstance(value, datetime):
                    redis_data[key] = value.isoformat()
                elif value is None:
                    redis_data[key] = ""
                else:
                    redis_data[key] = str(value)

            self.redis_client.hset(reading_key, mapping=redis_data)

            # Add to device's reading index (sorted by timestamp)
            device_readings_key = f"readings:{reading.device_id}"
            self.redis_client.zadd(
                device_readings_key,
                {reading_key: reading.timestamp.timestamp()}
            )

            # Keep only recent readings (last 1000)
            self.redis_client.zremrangebyrank(device_readings_key, 0, -1001)

            return True
        except Exception as e:
            logger.error(f"Error storing energy reading: {e}")
            return False

    def get_device_readings(
        self,
        device_id: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[EnergyReading]:
        """Get energy readings for a device"""
        try:
            device_readings_key = f"readings:{device_id}"

            # Set time range
            min_score = start_time.timestamp() if start_time else '-inf'
            max_score = end_time.timestamp() if end_time else '+inf'

            # Get reading keys in time order (most recent first)
            reading_keys = self.redis_client.zrevrangebyscore(
                device_readings_key,
                max_score,
                min_score,
                start=0,
                num=limit
            )

            readings = []
            for reading_key in reading_keys:
                reading_data = self.redis_client.hgetall(reading_key)
                if reading_data:
                    # Convert data back from Redis strings
                    parsed_data = {}
                    for key, value in reading_data.items():
                        if key == 'timestamp':
                            parsed_data[key] = datetime.fromisoformat(value)
                        elif key in ['energy_kwh', 'power_kw', 'voltage', 'current', 'power_factor', 'anomaly_score']:
                            try:
                                parsed_data[key] = float(value) if value and value != "" else None
                            except:
                                parsed_data[key] = None
                        elif key == 'is_anomaly':
                            parsed_data[key] = value.lower() == 'true'
                        else:
                            parsed_data[key] = value

                    readings.append(EnergyReading(**parsed_data))

            return readings
        except Exception as e:
            logger.error(f"Error retrieving readings for device {device_id}: {e}")
            return []

    def get_energy_readings(
        self,
        device_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[EnergyReading]:
        """Get energy readings for a device (alias for get_device_readings)"""
        return self.get_device_readings(device_id, limit, start_time, end_time)

    def store_embedding(self, content: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Store text embedding for semantic search"""
        try:
            # Convert numpy array to list for JSON serialization
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

            # Create document
            doc_id = f"doc:{hash(content)}"
            doc_data = {
                'content': content,
                'embedding': json.dumps(embedding_list),
                'metadata': json.dumps(metadata),
                'created_at': datetime.utcnow().isoformat()
            }

            self.redis_client.hset(doc_id, mapping=doc_data)
            self.redis_client.sadd("embeddings:index", doc_id)

            return True
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return False

    def semantic_search(self, query_embedding: np.ndarray, limit: int = 10) -> List[SemanticSearchResult]:
        """Perform semantic search using embeddings"""
        try:
            # Get all embedding documents
            doc_ids = self.redis_client.smembers("embeddings:index")
            results = []

            query_embedding = np.array(query_embedding)

            for doc_id in doc_ids:
                doc_data = self.redis_client.hgetall(doc_id)
                if not doc_data:
                    continue

                # Parse embedding
                stored_embedding = np.array(json.loads(doc_data['embedding']))

                # Calculate cosine similarity
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )

                metadata = json.loads(doc_data['metadata'])

                results.append(SemanticSearchResult(
                    device_id=metadata.get('device_id', 'unknown'),
                    content=doc_data['content'],
                    score=float(similarity),
                    metadata=metadata
                ))

            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.score, reverse=True)

            return results[:limit]
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis database statistics"""
        try:
            info = self.redis_client.info()
            device_count = self.redis_client.scard("devices:index")
            embedding_count = self.redis_client.scard("embeddings:index")

            return {
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "total_commands_processed": info.get("total_commands_processed"),
                "connected_clients": info.get("connected_clients"),
                "device_count": device_count,
                "embedding_count": embedding_count
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {}

# Global Redis service instance
redis_service = RedisService()
