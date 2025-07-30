#!/usr/bin/env python3
"""
Test script for Redis 8 vector search implementation
"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings
from app.services.redis_service import RedisService
from app.services.ai_service import AIService

def test_vector_implementation():
    """Test the vector search implementation"""
    print("🧪 Testing Redis 8 Vector Search Implementation")
    print("=" * 60)

    try:
        # Initialize services
        settings = Settings()
        redis_service = RedisService(settings)
        ai_service = AIService()

        print("✅ Services initialized")

        # Test embedding generation
        print("\n🧠 Testing embedding generation...")
        test_text = "HVAC system for climate control"
        embedding = ai_service.generate_embedding(test_text)
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        print(f"📊 Sample values: {embedding[:5]}...")

        # Test vector index creation
        print("\n🔧 Testing vector index creation...")
        index_success = redis_service.create_vector_index()
        if index_success:
            print("✅ Vector index created successfully")
        else:
            print("⚠️ Vector index creation failed or already exists")

        # Test device embedding generation
        print("\n📊 Testing device embedding generation...")
        devices = redis_service.get_all_devices()
        if devices:
            print(f"Found {len(devices)} devices")

            # Test embedding for first device
            device = devices[0]
            device_data = device.model_dump()
            content = ai_service.create_device_embedding(device_data)
            print(f"✅ Created searchable content: {content[:100]}...")

            # Generate all embeddings
            embedding_success = redis_service.generate_device_embeddings()
            if embedding_success:
                print("✅ All device embeddings generated")
            else:
                print("⚠️ Embedding generation had issues")
        else:
            print("⚠️ No devices found - seed data first")
            return

        # Test vector search
        print("\n🔍 Testing vector search...")
        test_queries = [
            "HVAC air conditioning",
            "energy efficient lighting",
            "server room cooling",
            "warehouse heating"
        ]

        for query in test_queries:
            print(f"\n🔍 Searching for: '{query}'")
            results = ai_service.semantic_search(query, limit=3)

            if results:
                print(f"✅ Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"   {i}. {result['device_id']}: {result['content'][:60]}... (score: {result['score']:.3f})")
            else:
                print("❌ No results found")

        print("\n🎉 Vector search implementation test completed!")
        print("\n💡 Try the CLI commands:")
        print("   uv run python cli.py vector-status")
        print("   uv run python cli.py search 'HVAC system'")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_implementation()
