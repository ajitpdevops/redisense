import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
import warnings
from config.settings import settings
from sentence_transformers import SentenceTransformer
import json

# Suppress specific PyTorch/Transformers warnings
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        self.embedding_model = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Initialize anomaly detector
            self.anomaly_detector = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42,
                n_estimators=100
            )
            logger.info("Initialized anomaly detection model")

            # Initialize embedding model with warning suppression
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*is deprecated.*")
                warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

                model_name = getattr(settings, 'EMBEDDING_MODEL', 'sentence-transformers/paraphrase-MiniLM-L6-v2')
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"Initialized embedding model: {model_name}")

        except Exception as e:
            logger.error(f"Error initializing AI models: {e}")
            # Fallback to mock embeddings if model fails to load
            self.embedding_model = None

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using sentence transformers"""
        try:
            if self.embedding_model is not None:
                # Use real sentence transformer with warning suppression
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*is deprecated.*")
                    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

                    embedding = self.embedding_model.encode(text)
                    return embedding.tolist()
            else:
                # Fallback to mock implementation
                return self._generate_mock_embedding(text)

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return self._generate_mock_embedding(text)

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding for text (fallback implementation)"""
        try:
            # For testing, return a simple hash-based embedding
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()

            # Convert hex to float values for 384-dim vector
            embedding = []
            for i in range(0, min(len(hash_hex), 96), 2):  # 96 hex chars = 48 bytes
                hex_pair = hash_hex[i:i+2]
                float_val = int(hex_pair, 16) / 255.0  # Normalize to 0-1
                embedding.extend([float_val] * 8)  # Repeat to fill 384 dims
                if len(embedding) >= settings.VECTOR_DIMENSION:
                    break

            # Pad or trim to exact dimension
            while len(embedding) < settings.VECTOR_DIMENSION:
                embedding.append(0.0)

            return embedding[:settings.VECTOR_DIMENSION]

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * settings.VECTOR_DIMENSION

    def detect_anomaly_simple(self, energy_value: float, device_history: List[float]) -> Tuple[bool, float]:
        """Simple Z-score based anomaly detection"""
        try:
            if len(device_history) < 3:  # Need minimum history
                return False, 0.0

            mean = np.mean(device_history)
            std = np.std(device_history)

            if std == 0:  # Avoid division by zero
                return False, 0.0

            z_score = abs((energy_value - mean) / std)
            is_anomaly = z_score > settings.ANOMALY_THRESHOLD

            return is_anomaly, z_score

        except Exception as e:
            logger.error(f"Error in simple anomaly detection: {e}")
            return False, 0.0

    def detect_anomaly_ml(self, energy_values: List[float]) -> Tuple[List[bool], List[float]]:
        """ML-based anomaly detection using Isolation Forest"""
        try:
            if len(energy_values) < 10:
                return [False] * len(energy_values), [0.0] * len(energy_values)

            # Prepare data
            X = np.array(energy_values).reshape(-1, 1)
            X_scaled = self.scaler.fit_transform(X)

            # Fit and predict
            self.anomaly_detector.fit(X_scaled)
            anomaly_labels = self.anomaly_detector.predict(X_scaled)
            anomaly_scores = self.anomaly_detector.decision_function(X_scaled)

            # Convert to boolean (IsolationForest returns -1 for anomalies, 1 for normal)
            is_anomalies = [label == -1 for label in anomaly_labels]

            # Normalize scores to 0-1 range
            normalized_scores = [(1 - score) / 2 for score in anomaly_scores]

            return is_anomalies, normalized_scores

        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {e}")
            return [False] * len(energy_values), [0.0] * len(energy_values)

    def analyze_device_pattern(self, device_id: str, energy_readings: List[Tuple[any, float]]) -> dict:
        """Analyze energy consumption patterns for a device"""
        try:
            if not energy_readings:
                return {"status": "insufficient_data"}

            values = [reading[1] for reading in energy_readings]

            # Basic statistics
            stats = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "total_readings": len(values)
            }

            # Anomaly analysis
            if len(values) >= 10:
                is_anomalies, scores = self.detect_anomaly_ml(values)
                anomaly_count = sum(is_anomalies)
                stats.update({
                    "anomaly_count": anomaly_count,
                    "anomaly_rate": anomaly_count / len(values),
                    "latest_anomaly_score": scores[-1] if scores else 0.0
                })

            # Pattern classification
            stats["pattern"] = self._classify_pattern(values)

            return stats

        except Exception as e:
            logger.error(f"Error analyzing device pattern: {e}")
            return {"status": "error", "message": str(e)}

    def _classify_pattern(self, values: List[float]) -> str:
        """Classify energy consumption pattern"""
        try:
            if len(values) < 5:
                return "insufficient_data"

            # Calculate trend
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]

            # Calculate variability
            cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0

            # Classification logic
            if abs(slope) < 0.01:
                if cv < 0.1:
                    return "stable"
                elif cv < 0.3:
                    return "moderate_variation"
                else:
                    return "high_variation"
            elif slope > 0.01:
                return "increasing"
            else:
                return "decreasing"

        except Exception as e:
            logger.error(f"Error classifying pattern: {e}")
            return "unknown"

    def generate_device_insights(self, device_data: dict, pattern_analysis: dict) -> str:
        """Generate human-readable insights about device behavior"""
        try:
            device_type = device_data.get("device_type", "Unknown")
            location = device_data.get("location", "Unknown location")

            insights = [
                f"Device Analysis for {device_type} in {location}:",
                f"- Average consumption: {pattern_analysis.get('mean', 0):.2f} kWh",
                f"- Consumption pattern: {pattern_analysis.get('pattern', 'unknown')}",
            ]

            if pattern_analysis.get("anomaly_count", 0) > 0:
                anomaly_rate = pattern_analysis.get("anomaly_rate", 0) * 100
                insights.append(f"- Anomaly rate: {anomaly_rate:.1f}%")

                if anomaly_rate > 20:
                    insights.append("- HIGH ALERT: Frequent anomalies detected!")
                elif anomaly_rate > 10:
                    insights.append("- Warning: Elevated anomaly rate")

            return "\n".join(insights)

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return "Unable to generate insights"

    def analyze_consumption_pattern(self, device_id: str, minutes: int) -> dict:
        """Analyze consumption pattern for CLI compatibility"""
        try:
            # This is a wrapper around analyze_device_pattern for CLI compatibility
            # In a real implementation, this would fetch data from Redis
            return {
                "pattern": "stable",
                "trend": "normal",
                "avg_consumption": 2.5,
                "status": "simulated_data"
            }
        except Exception as e:
            logger.error(f"Error in consumption pattern analysis: {e}")
            return {"pattern": "unknown", "trend": "unknown", "avg_consumption": 0}

    def detect_anomalies(self, device_id: str, minutes: int) -> List[dict]:
        """Detect anomalies for CLI compatibility"""
        try:
            # This is a placeholder implementation for CLI compatibility
            # In a real implementation, this would fetch and analyze data from Redis
            return []  # No anomalies for now
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return []

    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Semantic search using embeddings and Redis vector search"""
        try:
            from app.services.redis_service import RedisService

            # Generate embedding for query
            query_embedding = self.generate_embedding(query)

            # Use Redis service for vector search
            redis_service = RedisService()
            results = redis_service.vector_search(query_embedding, limit)

            # Convert to dictionary format for CLI compatibility
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'device_id': result.device_id,
                    'content': result.content,
                    'score': result.score,
                    'metadata': result.metadata
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def create_device_embedding(self, device_data: Dict[str, Any]) -> str:
        """Create searchable text content from device data"""
        try:
            # Combine device information into searchable text
            parts = []

            if device_data.get('name'):
                parts.append(f"Device: {device_data['name']}")

            if device_data.get('device_type'):
                parts.append(f"Type: {device_data['device_type']}")

            if device_data.get('manufacturer'):
                parts.append(f"Manufacturer: {device_data['manufacturer']}")

            if device_data.get('model'):
                parts.append(f"Model: {device_data['model']}")

            if device_data.get('location'):
                parts.append(f"Location: {device_data['location']}")

            if device_data.get('description'):
                parts.append(f"Description: {device_data['description']}")

            # Add power rating and efficiency info
            if device_data.get('power_rating'):
                parts.append(f"Power Rating: {device_data['power_rating']}W")

            # Add metadata if available
            metadata = device_data.get('metadata', {})
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    if key == 'energy_efficiency_rating':
                        parts.append(f"Energy Efficiency: {value}")
                    elif key == 'maintenance_schedule':
                        parts.append(f"Maintenance: {value}")
                    elif key == 'operating_hours':
                        parts.append(f"Operating Hours: {value}")

            return " | ".join(parts)

        except Exception as e:
            logger.error(f"Error creating device embedding text: {e}")
            return device_data.get('name', 'Unknown Device')

# Global AI service instance
ai_service = AIService()
