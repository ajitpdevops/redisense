"""
Unit tests for configuration
"""
import pytest
import os
from unittest.mock import patch
from pydantic import ValidationError

from config.settings import Settings


class TestSettings:
    """Test Settings configuration"""

    @patch.dict(os.environ, {}, clear=True)  # Clear all env vars for clean test
    def test_default_settings(self):
        """Test default configuration values"""
        settings = Settings()

        assert settings.REDIS_HOST == "localhost"
        assert settings.REDIS_PORT == 6379
        assert settings.REDIS_USERNAME == ""
        assert settings.REDIS_PASSWORD == ""
        assert settings.DEBUG is False  # Default is False
        assert settings.LOG_LEVEL == "INFO"
        assert settings.ANOMALY_THRESHOLD == 2.0
        assert settings.DEVICE_COUNT == 5
        assert settings.DATA_GENERATION_INTERVAL == 60
        assert settings.ENERGY_MIN == 0.5
        assert settings.ENERGY_MAX == 10.0
        assert settings.ANOMALY_SPIKE_MULTIPLIER == 3.0

    @patch.dict(os.environ, {
        'REDIS_HOST': 'test-redis.com',
        'REDIS_PORT': '12345',
        'REDIS_USERNAME': 'testuser',
        'REDIS_PASSWORD': 'testpass',
        'DEBUG': 'false',
        'LOG_LEVEL': 'DEBUG',
        'ANOMALY_THRESHOLD': '1.5',
        'DEVICE_COUNT': '10',
        'DATA_GENERATION_INTERVAL': '30',
        'ENERGY_MIN': '1.0',
        'ENERGY_MAX': '20.0',
        'ANOMALY_SPIKE_MULTIPLIER': '5.0'
    }, clear=True)
    def test_environment_overrides(self):
        """Test that environment variables override defaults"""
        settings = Settings()

        assert settings.REDIS_HOST == "test-redis.com"
        assert settings.REDIS_PORT == 12345
        assert settings.REDIS_USERNAME == "testuser"
        assert settings.REDIS_PASSWORD == "testpass"
        assert settings.DEBUG is False
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.ANOMALY_THRESHOLD == 1.5
        assert settings.DEVICE_COUNT == 10
        assert settings.DATA_GENERATION_INTERVAL == 30
        assert settings.ENERGY_MIN == 1.0
        assert settings.ENERGY_MAX == 20.0
        assert settings.ANOMALY_SPIKE_MULTIPLIER == 5.0

    @patch.dict(os.environ, {'DEBUG': 'True'}, clear=True)
    def test_debug_true_variations(self):
        """Test DEBUG accepts various true values"""
        settings = Settings()
        assert settings.DEBUG is True

    @patch.dict(os.environ, {'DEBUG': 'FALSE'}, clear=True)
    def test_debug_false_variations(self):
        """Test DEBUG accepts various false values"""
        settings = Settings()
        assert settings.DEBUG is False

    @patch.dict(os.environ, {'REDIS_PORT': 'invalid'}, clear=True)
    def test_invalid_port_raises_error(self):
        """Test invalid port number raises error"""
        with pytest.raises(ValidationError):
            Settings()

    @patch.dict(os.environ, {'ANOMALY_THRESHOLD': 'invalid'}, clear=True)
    def test_invalid_float_raises_error(self):
        """Test invalid float value raises error"""
        with pytest.raises(ValidationError):
            Settings()

    @patch.dict(os.environ, {'DEVICE_COUNT': 'invalid'}, clear=True)
    def test_invalid_int_raises_error(self):
        """Test invalid integer value raises error"""
        with pytest.raises(ValidationError):
            Settings()

    @patch.dict(os.environ, {
        'REDIS_HOST': 'redis-12931.c1.ap-southeast-1-1.ec2.redns.redis-cloud.com',
        'REDIS_PORT': '12931',
        'REDIS_USERNAME': 'username_here',
        'REDIS_PASSWORD': 'password_here'
    }, clear=True)
    def test_redis_cloud_format(self):
        """Test Redis Cloud connection format"""
        settings = Settings()

        assert "redis-cloud.com" in settings.REDIS_HOST
        assert settings.REDIS_PORT == 12931
        assert settings.REDIS_USERNAME == "username_here"
        assert settings.REDIS_PASSWORD == "password_here"

    @patch.dict(os.environ, {}, clear=True)
    def test_embedding_model_default(self):
        """Test default embedding model"""
        settings = Settings()
        assert "sentence-transformers" in settings.EMBEDDING_MODEL
        assert "paraphrase-MiniLM-L6-v2" in settings.EMBEDDING_MODEL

    @patch.dict(os.environ, {}, clear=True)
    def test_vector_dimension_default(self):
        """Test default vector dimension for MiniLM"""
        settings = Settings()
        assert settings.VECTOR_DIMENSION == 384  # MiniLM dimension

    @patch.dict(os.environ, {
        'EMBEDDING_MODEL': 'custom-model',
        'VECTOR_DIMENSION': '512'
    }, clear=True)
    def test_custom_embedding_config(self):
        """Test custom embedding configuration"""
        settings = Settings()
        assert settings.EMBEDDING_MODEL == "custom-model"
        assert settings.VECTOR_DIMENSION == 512
