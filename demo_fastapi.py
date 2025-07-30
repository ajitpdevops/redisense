#!/usr/bin/env python3
"""
Demo script to showcase the FastAPI interface vs Streamlit
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_banner():
    """Print a nice banner"""
    print("=" * 80)
    print("🌟 REDISENSE FASTAPI WEB INTERFACE DEMO 🌟")
    print("=" * 80)
    print()
    print("🚀 A Modern Alternative to Streamlit")
    print("📱 Responsive, Fast, and Production-Ready")
    print("⚡ Powered by FastAPI + Bootstrap + Chart.js")
    print()

def show_features():
    """Show the key features"""
    features = [
        "🎨 Modern Glass-morphism Design",
        "📊 Real-time Metrics with Auto-refresh",
        "🔍 AI-Powered Semantic Search",
        "📱 Mobile-First Responsive Design",
        "⚡ Lightning-fast Performance",
        "🎯 Professional UI/UX",
        "🛡️ Production-Ready Architecture",
        "📈 Interactive Charts & Analytics",
        "🔧 Highly Customizable",
        "🌙 Dark Mode Support"
    ]

    print("✨ KEY FEATURES:")
    print("-" * 50)
    for feature in features:
        print(f"  {feature}")
        time.sleep(0.1)
    print()

def show_comparison():
    """Show Streamlit vs FastAPI comparison"""
    print("🆚 STREAMLIT vs FASTAPI COMPARISON:")
    print("-" * 50)
    comparison = [
        ("Setup Complexity", "⭐⭐⭐⭐⭐ Simple", "⭐⭐⭐ Moderate"),
        ("Performance", "⭐⭐ Slower", "⭐⭐⭐⭐⭐ Fast"),
        ("Customization", "⭐⭐ Limited", "⭐⭐⭐⭐⭐ Unlimited"),
        ("Mobile Support", "⭐⭐ Basic", "⭐⭐⭐⭐⭐ Excellent"),
        ("Production Ready", "⭐⭐ Okay", "⭐⭐⭐⭐⭐ Perfect"),
        ("UI Control", "⭐⭐ Constrained", "⭐⭐⭐⭐⭐ Full Control"),
        ("Load Times", "⭐⭐ Slower", "⭐⭐⭐⭐⭐ Instant"),
        ("Scalability", "⭐⭐ Limited", "⭐⭐⭐⭐⭐ Excellent")
    ]

    for metric, streamlit, fastapi in comparison:
        print(f"  {metric:15} | {streamlit:20} | {fastapi}")
        time.sleep(0.1)
    print()

def show_file_structure():
    """Show the file structure"""
    print("📁 FASTAPI PROJECT STRUCTURE:")
    print("-" * 50)
    structure = [
        "web_app.py              # Main FastAPI application",
        "web/",
        "├── templates/",
        "│   ├── base.html       # Base template with navigation",
        "│   ├── dashboard.html  # Main dashboard",
        "│   ├── devices.html    # Device management",
        "│   ├── analytics.html  # Charts & analytics",
        "│   └── search.html     # Semantic search",
        "├── static/",
        "│   ├── css/",
        "│   │   └── dashboard.css # Modern styling",
        "│   └── js/",
        "│       └── dashboard.js  # Interactive features",
        "start_web.sh           # Easy startup script",
        "FASTAPI_README.md      # Complete documentation"
    ]

    for item in structure:
        print(f"  {item}")
        time.sleep(0.05)
    print()

def show_instructions():
    """Show how to start the application"""
    print("🚀 HOW TO START THE APPLICATION:")
    print("-" * 50)
    print("  1. Quick Start:")
    print("     ./start_web.sh")
    print()
    print("  2. Direct Command:")
    print("     uv run uvicorn web_app:app --host 0.0.0.0 --port 8080 --reload")
    print()
    print("  3. Python Script:")
    print("     uv run python web_app.py")
    print()
    print("  4. Then open: http://localhost:8080")
    print()

def show_pages():
    """Show available pages"""
    print("📄 AVAILABLE PAGES:")
    print("-" * 50)
    pages = [
        ("🏠 Dashboard", "/", "Real-time metrics, system overview"),
        ("💻 Devices", "/devices", "Device management & monitoring"),
        ("📊 Analytics", "/analytics", "Charts, trends, performance"),
        ("🔍 Search", "/search", "AI-powered semantic search"),
        ("🔧 Device Detail", "/devices/{id}", "Individual device analysis")
    ]

    for icon_name, url, description in pages:
        print(f"  {icon_name:12} {url:15} - {description}")
        time.sleep(0.1)
    print()

def main():
    """Main demo function"""
    print_banner()
    show_features()
    show_comparison()
    show_file_structure()
    show_pages()
    show_instructions()

    print("💡 TIP: The FastAPI interface provides a more professional,")
    print("    performant, and customizable experience compared to Streamlit!")
    print()
    print("🎉 Ready to replace Streamlit with a modern web interface!")
    print("=" * 80)

if __name__ == "__main__":
    main()
