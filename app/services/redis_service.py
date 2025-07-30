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

    def create_vector_index(self) -> bool:
        """Create Redis vector index for device embeddings using Redis 8 capabilities"""
        try:
            # Check if index already exists
            try:
                self.redis_client.ft("device_embeddings").info()
                logger.info("Vector index 'device_embeddings' already exists")
                return True
            except:
                # Index doesn't exist, create it
                pass

            # Try different import methods for Redis search
            try:
                # Method 1: Modern redis-py imports
                from redis.commands.search.field import VectorField, TextField, TagField
                from redis.commands.search.indexDefinition import IndexDefinition, IndexType
                use_modern_api = True
            except ImportError:
                try:
                    # Method 2: Alternative imports
                    from redisearch import VectorField, TextField, TagField, IndexDefinition
                    use_modern_api = False
                except ImportError:
                    # Method 3: Manual command construction
                    logger.warning("RediSearch Python modules not available, using manual commands")
                    return self._create_vector_index_manual()

            if use_modern_api:
                # Create vector index schema using modern API
                schema = [
                    TextField("content"),
                    TagField("device_id"),
                    TagField("device_type"),
                    TagField("manufacturer"),
                    TagField("location"),
                    VectorField(
                        "embedding",
                        "FLAT",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": 384,  # sentence-transformers/paraphrase-MiniLM-L6-v2 dimension
                            "DISTANCE_METRIC": "COSINE"
                        }
                    )
                ]

                # Create index
                self.redis_client.ft("device_embeddings").create_index(
                    schema,
                    definition=IndexDefinition(
                        prefix=["device_embed:"],
                        index_type=IndexType.HASH
                    )
                )
            else:
                # Use alternative API
                schema = [
                    TextField("content"),
                    TagField("device_id"),
                    TagField("device_type"),
                    TagField("manufacturer"),
                    TagField("location"),
                    VectorField(
                        "embedding",
                        "FLAT",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": 384,
                            "DISTANCE_METRIC": "COSINE"
                        }
                    )
                ]

                index_def = IndexDefinition(prefix=["device_embed:"])
                self.redis_client.ft("device_embeddings").create_index(schema, definition=index_def)

            logger.info("Created Redis vector index 'device_embeddings'")
            return True

        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            # Try manual method as fallback
            return self._create_vector_index_manual()

    def _create_vector_index_manual(self) -> bool:
        """Create vector index using manual Redis commands"""
        try:
            # Check if index exists
            try:
                self.redis_client.execute_command("FT.INFO", "device_embeddings")
                logger.info("Vector index 'device_embeddings' already exists")
                return True
            except:
                pass

            # Create index using raw Redis commands
            cmd = [
                "FT.CREATE", "device_embeddings",
                "ON", "HASH",
                "PREFIX", "1", "device_embed:",
                "SCHEMA",
                "content", "TEXT",
                "device_id", "TAG",
                "device_type", "TAG",
                "manufacturer", "TAG",
                "location", "TAG",
                "embedding", "VECTOR", "FLAT", "6",
                "TYPE", "FLOAT32",
                "DIM", "384",
                "DISTANCE_METRIC", "COSINE"
            ]

            self.redis_client.execute_command(*cmd)
            logger.info("Created Redis vector index 'device_embeddings' using manual commands")
            return True

        except Exception as e:
            logger.error(f"Error creating vector index manually: {e}")
            return False

    def store_device_embedding(self, device_id: str, content: str, embedding: List[float], metadata: dict) -> bool:
        """Store device embedding using Redis 8 vector index"""
        try:
            # Convert embedding to bytes for Redis storage
            import struct
            embedding_bytes = struct.pack(f'{len(embedding)}f', *embedding)

            # Create document for vector index
            doc_key = f"device_embed:{device_id}"
            doc_data = {
                "content": content,
                "device_id": device_id,
                "device_type": metadata.get("device_type", ""),
                "manufacturer": metadata.get("manufacturer", ""),
                "location": metadata.get("location", ""),
                "embedding": embedding_bytes,
                "created_at": datetime.utcnow().isoformat()
            }

            # Store document
            self.redis_client.hset(doc_key, mapping=doc_data)

            logger.info(f"Stored device embedding for {device_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing device embedding: {e}")
            return False

    def vector_search(self, query_embedding: List[float], limit: int = 10) -> List[SemanticSearchResult]:
        """Perform vector similarity search using Redis 8 vector capabilities"""
        try:
            # Convert query embedding to bytes
            import struct
            query_bytes = struct.pack(f'{len(query_embedding)}f', *query_embedding)

            # Perform KNN search using Redis FT.SEARCH
            query = f"*=>[KNN {limit} @embedding $query_vector AS score]"

            results = self.redis_client.ft("device_embeddings").search(
                query,
                query_params={"query_vector": query_bytes}
            )

            # Convert results to SemanticSearchResult objects
            search_results = []
            for doc in results.docs:
                try:
                    search_results.append(SemanticSearchResult(
                        device_id=doc.device_id,
                        content=doc.content,
                        score=float(doc.score),
                        metadata={
                            "device_type": getattr(doc, 'device_type', ''),
                            "manufacturer": getattr(doc, 'manufacturer', ''),
                            "location": getattr(doc, 'location', '')
                        }
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing search result: {e}")
                    continue

            return search_results

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            # Fallback to old semantic search method
            return self.semantic_search(np.array(query_embedding), limit)

    def generate_device_embeddings(self) -> bool:
        """Generate embeddings for all existing devices"""
        try:
            from app.services.ai_service import AIService

            ai_service = AIService()
            devices = self.get_all_devices()

            if not devices:
                logger.warning("No devices found to generate embeddings for")
                return False

            # Ensure vector index exists
            self.create_vector_index()

            success_count = 0
            for device in devices:
                try:
                    # Convert device to searchable text
                    device_data = device.model_dump()
                    content = ai_service.create_device_embedding(device_data)

                    # Generate embedding
                    embedding = ai_service.generate_embedding(content)

                    # Store in vector index
                    metadata = {
                        "device_type": device.device_type.value if hasattr(device.device_type, 'value') else str(device.device_type),
                        "manufacturer": getattr(device, 'manufacturer', ''),
                        "location": getattr(device, 'location', ''),
                        "model": getattr(device, 'model', ''),
                        "power_rating": getattr(device, 'power_rating', 0)
                    }

                    success = self.store_device_embedding(device.device_id, content, embedding, metadata)
                    if success:
                        success_count += 1

                except Exception as e:
                    logger.error(f"Error generating embedding for device {device.device_id}: {e}")
                    continue

            logger.info(f"Generated embeddings for {success_count}/{len(devices)} devices")
            return success_count > 0

        except Exception as e:
            logger.error(f"Error generating device embeddings: {e}")
            return False

    def clear_energy_readings(self, device_id: Optional[str] = None) -> int:
        """Clear energy readings for a specific device or all devices"""
        try:
            if device_id:
                # Clear readings for specific device
                reading_key = f"readings:{device_id}"
                keys_deleted = self.redis_client.delete(reading_key)
                logger.info(f"Cleared {keys_deleted} reading sets for device {device_id}")
                return keys_deleted
            else:
                # Clear all energy readings
                reading_pattern = "readings:*"
                keys = self.redis_client.keys(reading_pattern)
                if keys:
                    keys_deleted = self.redis_client.delete(*keys)
                    logger.info(f"Cleared {keys_deleted} reading sets")
                    return keys_deleted
                return 0
        except Exception as e:
            logger.error(f"Error clearing energy readings: {e}")
            return 0

    def clear_all_devices(self) -> int:
        """Clear all device data"""
        try:
            device_pattern = "device:*"
            keys = self.redis_client.keys(device_pattern)
            if keys:
                keys_deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {keys_deleted} devices")
                return keys_deleted
            return 0
        except Exception as e:
            logger.error(f"Error clearing devices: {e}")
            return 0

    def clear_all_data(self) -> Dict[str, int]:
        """Clear all data from Redis"""
        try:
            # Get counts before clearing
            device_keys = self.redis_client.keys("device:*")
            reading_keys = self.redis_client.keys("readings:*")
            vector_keys = self.redis_client.keys("vector:*")

            all_keys = device_keys + reading_keys + vector_keys

            if all_keys:
                keys_deleted = self.redis_client.delete(*all_keys)
                logger.info(f"Cleared all data: {keys_deleted} keys deleted")
                return {
                    "devices": len(device_keys),
                    "readings": len(reading_keys),
                    "vectors": len(vector_keys),
                    "total": keys_deleted
                }
            return {"devices": 0, "readings": 0, "vectors": 0, "total": 0}
        except Exception as e:
            logger.error(f"Error clearing all data: {e}")
            return {"devices": 0, "readings": 0, "vectors": 0, "total": 0}

    def get_data_statistics(self) -> Dict[str, int]:
        """Get statistics about data in Redis"""
        try:
            device_count = len(self.redis_client.keys("device:*"))
            reading_count = len(self.redis_client.keys("readings:*"))
            vector_count = len(self.redis_client.keys("vector:*"))

            # Estimate total readings by sampling
            total_readings = 0
            sample_keys = self.redis_client.keys("readings:*")[:5]  # Sample first 5
            for key in sample_keys:
                readings = self.redis_client.llen(key)
                total_readings += readings

            # Extrapolate based on sample
            if sample_keys and reading_count > 0:
                avg_readings_per_device = total_readings / len(sample_keys)
                estimated_total_readings = int(avg_readings_per_device * reading_count)
            else:
                estimated_total_readings = 0

            return {
                "devices": device_count,
                "reading_keys": reading_count,
                "estimated_total_readings": estimated_total_readings,
                "vectors": vector_count
            }
        except Exception as e:
            logger.error(f"Error getting data statistics: {e}")
            return {"devices": 0, "reading_keys": 0, "estimated_total_readings": 0, "vectors": 0}

# Global Redis service instance
redis_service = RedisService()
