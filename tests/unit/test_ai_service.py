"""
Unit tests for AI Service
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.services.ai_service import AIService


class TestAIService:
    """Test AI Service functionality"""

    @pytest.fixture
    def ai_service(self):
        """Create AI service instance for testing"""
        return AIService()

    def test_ai_service_initialization(self, ai_service):
        """Test AI service initializes correctly"""
        assert ai_service.scaler is not None
        assert ai_service.anomaly_detector is not None

    def test_generate_embedding_basic(self, ai_service):
        """Test embedding generation returns correct format"""
        text = "HVAC energy consumption anomaly"
        embedding = ai_service.generate_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384  # Default dimension
        assert all(isinstance(x, float) for x in embedding)
        assert all(0.0 <= x <= 1.0 for x in embedding)  # Normalized values

    def test_generate_embedding_consistent(self, ai_service):
        """Test same text produces same embedding"""
        text = "test energy reading"
        embedding1 = ai_service.generate_embedding(text)
        embedding2 = ai_service.generate_embedding(text)

        assert embedding1 == embedding2

    def test_generate_embedding_different_texts(self, ai_service):
        """Test different texts produce different embeddings"""
        text1 = "HVAC system normal"
        text2 = "Server anomaly detected"

        embedding1 = ai_service.generate_embedding(text1)
        embedding2 = ai_service.generate_embedding(text2)

        assert embedding1 != embedding2

    def test_generate_embedding_empty_text(self, ai_service):
        """Test embedding generation with empty text"""
        embedding = ai_service.generate_embedding("")

        assert isinstance(embedding, list)
        assert len(embedding) == 384

    def test_detect_anomaly_simple_insufficient_data(self, ai_service):
        """Test anomaly detection with insufficient historical data"""
        energy_value = 5.0
        device_history = [4.0, 5.0]  # Less than minimum required

        is_anomaly, score = ai_service.detect_anomaly_simple(energy_value, device_history)

        assert is_anomaly is False
        assert score == 0.0

    def test_detect_anomaly_simple_normal_value(self, ai_service):
        """Test anomaly detection with normal energy value"""
        energy_value = 5.0
        device_history = [4.5, 5.0, 5.5, 4.8, 5.2, 4.9, 5.1]  # Stable history

        is_anomaly, score = ai_service.detect_anomaly_simple(energy_value, device_history)

        assert bool(is_anomaly) is False
        assert isinstance(score, float)
        assert score >= 0.0

    def test_detect_anomaly_simple_anomalous_value(self, ai_service):
        """Test anomaly detection with anomalous energy value"""
        energy_value = 15.0  # Much higher than history
        device_history = [4.5, 5.0, 5.5, 4.8, 5.2, 4.9, 5.1]

        is_anomaly, score = ai_service.detect_anomaly_simple(energy_value, device_history)

        assert bool(is_anomaly) is True
        assert isinstance(score, float)
        assert score > 2.0  # Should exceed threshold

    def test_detect_anomaly_simple_zero_std(self, ai_service):
        """Test anomaly detection with zero standard deviation"""
        energy_value = 5.0
        device_history = [5.0, 5.0, 5.0, 5.0]  # All same values

        is_anomaly, score = ai_service.detect_anomaly_simple(energy_value, device_history)

        assert is_anomaly is False
        assert score == 0.0

    def test_detect_anomaly_ml_insufficient_data(self, ai_service):
        """Test ML anomaly detection with insufficient data"""
        energy_values = [1.0, 2.0, 3.0]  # Less than 10 values

        is_anomalies, scores = ai_service.detect_anomaly_ml(energy_values)

        assert len(is_anomalies) == len(energy_values)
        assert len(scores) == len(energy_values)
        assert all(anomaly is False for anomaly in is_anomalies)
        assert all(score == 0.0 for score in scores)

    def test_detect_anomaly_ml_normal_data(self, ai_service):
        """Test ML anomaly detection with normal data"""
        # Generate normal-looking data
        np.random.seed(42)
        energy_values = np.random.normal(5.0, 0.5, 20).tolist()

        is_anomalies, scores = ai_service.detect_anomaly_ml(energy_values)

        assert len(is_anomalies) == len(energy_values)
        assert len(scores) == len(energy_values)
        assert isinstance(is_anomalies, list)
        assert isinstance(scores, list)
        assert all(isinstance(score, float) for score in scores)

    def test_detect_anomaly_ml_with_outliers(self, ai_service):
        """Test ML anomaly detection with clear outliers"""
        # Normal data with clear outliers
        energy_values = [5.0] * 15 + [25.0, 30.0]  # Clear outliers at the end

        is_anomalies, scores = ai_service.detect_anomaly_ml(energy_values)

        assert len(is_anomalies) == len(energy_values)
        assert len(scores) == len(energy_values)
        # At least some values should be detected as anomalies
        assert sum(is_anomalies) > 0

    def test_analyze_device_pattern_empty_data(self, ai_service):
        """Test pattern analysis with no data"""
        result = ai_service.analyze_device_pattern("device_001", [])

        assert result["status"] == "insufficient_data"

    def test_analyze_device_pattern_basic_stats(self, ai_service):
        """Test pattern analysis calculates basic statistics"""
        # Create sample readings (timestamp, energy_kwh)
        readings = [
            (None, 4.0), (None, 5.0), (None, 6.0), (None, 5.5), (None, 4.5)
        ]

        result = ai_service.analyze_device_pattern("device_001", readings)

        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert "median" in result
        assert "total_readings" in result
        assert "pattern" in result

        assert result["total_readings"] == 5
        assert result["min"] == 4.0
        assert result["max"] == 6.0
        assert 4.5 <= result["mean"] <= 5.5

    def test_analyze_device_pattern_with_anomalies(self, ai_service):
        """Test pattern analysis with enough data for anomaly detection"""
        # Create 15 readings - enough for ML analysis
        normal_values = [5.0 + 0.1 * i for i in range(12)]  # Slightly increasing
        anomaly_values = [15.0, 20.0, 18.0]  # Clear anomalies
        all_values = normal_values + anomaly_values

        readings = [(None, value) for value in all_values]

        result = ai_service.analyze_device_pattern("device_001", readings)

        assert "anomaly_count" in result
        assert "anomaly_rate" in result
        assert "latest_anomaly_score" in result
        assert result["total_readings"] == 15
        assert result["anomaly_count"] >= 0
        assert 0.0 <= result["anomaly_rate"] <= 1.0

    def test_classify_pattern_insufficient_data(self, ai_service):
        """Test pattern classification with insufficient data"""
        values = [1.0, 2.0]  # Less than 5 values

        pattern = ai_service._classify_pattern(values)

        assert pattern == "insufficient_data"

    def test_classify_pattern_stable(self, ai_service):
        """Test pattern classification for stable consumption"""
        values = [5.0, 5.1, 4.9, 5.0, 5.1, 4.95, 5.05]  # Very stable

        pattern = ai_service._classify_pattern(values)

        assert pattern in ["stable", "moderate_variation"]

    def test_classify_pattern_increasing(self, ai_service):
        """Test pattern classification for increasing consumption"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # Clear increase

        pattern = ai_service._classify_pattern(values)

        assert pattern == "increasing"

    def test_classify_pattern_decreasing(self, ai_service):
        """Test pattern classification for decreasing consumption"""
        values = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]  # Clear decrease

        pattern = ai_service._classify_pattern(values)

        assert pattern == "decreasing"

    def test_classify_pattern_high_variation(self, ai_service):
        """Test pattern classification for high variation"""
        values = [5.0, 12.0, 4.0, 11.0, 5.5, 13.0, 4.5, 12.5, 5.2]  # High variation, no clear trend

        pattern = ai_service._classify_pattern(values)

        # Should be high_variation since CV > 0.3 and slope is minimal
        assert pattern in ["high_variation", "moderate_variation"]  # Accept either high or moderate variation

    def test_generate_device_insights_basic(self, ai_service):
        """Test device insights generation"""
        device_data = {
            "device_type": "HVAC",
            "location": "Building A"
        }
        pattern_analysis = {
            "mean": 5.2,
            "pattern": "stable",
            "anomaly_count": 0
        }

        insights = ai_service.generate_device_insights(device_data, pattern_analysis)

        assert isinstance(insights, str)
        assert "HVAC" in insights
        assert "Building A" in insights
        assert "5.2" in insights
        assert "stable" in insights

    def test_generate_device_insights_with_anomalies(self, ai_service):
        """Test device insights with anomalies"""
        device_data = {
            "device_type": "Server",
            "location": "Data Center"
        }
        pattern_analysis = {
            "mean": 8.5,
            "pattern": "high_variation",
            "anomaly_count": 5,
            "anomaly_rate": 0.25  # 25% anomaly rate
        }

        insights = ai_service.generate_device_insights(device_data, pattern_analysis)

        assert "Server" in insights
        assert "Data Center" in insights
        assert "25.0%" in insights
        assert "HIGH ALERT" in insights  # Should trigger high alert

    def test_generate_device_insights_error_handling(self, ai_service):
        """Test insights generation with missing data"""
        device_data = {}  # Empty device data
        pattern_analysis = {}  # Empty analysis

        insights = ai_service.generate_device_insights(device_data, pattern_analysis)

        assert isinstance(insights, str)
        assert "Unknown" in insights  # Should handle missing data gracefully

    @patch('app.services.ai_service.logger')
    def test_error_handling_in_anomaly_detection(self, mock_logger, ai_service):
        """Test error handling in anomaly detection"""
        # Force an error by passing invalid data
        with patch.object(np, 'mean', side_effect=Exception("Test error")):
            is_anomaly, score = ai_service.detect_anomaly_simple(5.0, [1.0, 2.0, 3.0])

            assert is_anomaly is False
            assert score == 0.0
            mock_logger.error.assert_called_once()

    @patch('app.services.ai_service.logger')
    def test_error_handling_in_pattern_analysis(self, mock_logger, ai_service):
        """Test error handling in pattern analysis"""
        # Force an error by patching numpy functions
        with patch.object(np, 'mean', side_effect=Exception("Test error")):
            result = ai_service.analyze_device_pattern("device_001", [(None, 5.0)])

            assert result["status"] == "error"
            assert "message" in result
            mock_logger.error.assert_called_once()
