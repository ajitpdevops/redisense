# 📄 SPECIFICATION.md

## 🚀 Project Title: Redisense

> Smart Energy Usage Monitoring and Anomaly Detection using Redis 8 + AI

---

## 🧩 Overview

**Redisense** is an AI-powered, real-time energy analytics and anomaly detection platform built with Redis 8 and FastAPI. It simulates energy consumption data, detects abnormal patterns, enables semantic search, and visualizes insights — all using Redis's latest features.

Built as part of the **Redis 8 AI Innovators Challenge**, it showcases:

- Real-time data ingestion with RedisTimeSeries
- Lightweight anomaly detection using AI/ML
- Semantic search with vector embeddings
- RedisJSON-based device metadata
- Rich querying with Redis Search
- Probabilistic data structures for top-K insights

---

## 🛠️ Tech Stack

|------------------|--------------------------------|
| Layer | Tech/Tool |
|------------------|--------------------------------|
| Backend API | Python + FastAPI |
| AI Inference | Scikit-learn + RedisAI |
| Embeddings | suitable AI Library |
| Vector Search | Redis 8 (HNSW Vector Index) |
| Time Series | RedisTimeSeries |
| Device Metadata | RedisJSON |
| Search Engine | RedisSearch (FT.SEARCH) |
| Probabilistic | RedisBloom TopK / Count-Min |
| Frontend | FastAPI + Jinja2 Templates |
| Dev & Sim Tools | Faker, Schedule, Requests |
|------------------|--------------------------------|

---

## 🎯 Core Features

### 1. 📡 **Energy Usage Ingestion**

- Simulated IoT-style energy data pushed every few minutes using `TS.ADD`
- Data retention and labels supported

### 2. 🧠 **Anomaly Detection**

- Predictive model (Regression or Isolation Forest) determines if reading is anomalous
- RedisAI support (optional)
- Detected anomalies are flagged and stored

### 3. 🧾 **Device Metadata Storage**

- Device info (type, location, install date) stored in RedisJSON
- Searchable via RedisSearch

### 4. 🔍 **Semantic Search**

- Natural language queries converted to embeddings via MiniLM
- Redis vector similarity search for pattern/event lookup

### 5. 📊 **Web Interface (FastAPI + Jinja2)**

- Modern responsive web interface with Bootstrap 5
- Visualizes live time series data
- Lists current anomalies with device details
- Supports semantic search functionality
- Shows top devices by activity or anomaly rate
- Real-time monitoring dashboard

### 6. 📈 **Query Engine**

- Query devices by metadata: location, type, anomaly status
- Example: `FT.SEARCH idx "@type:HVAC @status:anomaly"`

### 7. 📦 **Probabilistic Metrics**

- Track frequently anomalous devices using RedisBloom TopK
- Detect heavy hitters (Count-Min Sketch)

---

## 📁 Project Structure

redisense/
├── app/
│ ├── main.py
│ ├── routes/ # FastAPI route modules
│ ├── services/ # Redis, AI, embedding logic
│ ├── models/ # Pydantic models
│ └── utils/ # Embedding, anomaly logic
├── web/
│ ├── templates/ # Jinja2 HTML templates
│ └── static/ # CSS, JS, images
├── data/
│ └── generator.py # Simulates incoming energy data
├── web_app.py # FastAPI web application
├── requirements.txt
└── SPECIFICATION.md

---

## 🔧 API Endpoints

| Endpoint           | Method | Description                        |
| ------------------ | ------ | ---------------------------------- |
| `/devices/`        | POST   | Register a new device              |
| `/energy/ingest`   | POST   | Add energy reading for a device    |
| `/anomalies/`      | GET    | List devices with recent anomalies |
| `/search/semantic` | POST   | Search logs/patterns via embedding |
| `/devices/{id}`    | GET    | Get metadata and recent usage      |

---

## 🧠 AI & Embedding Details

- **Anomaly Detection**

  - Option 1: Z-score or rule-based thresholds
  - Option 2: Scikit-learn (IsolationForest, OneClassSVM)
  - Optional: Load model into RedisAI for edge inference

- **Embeddings**
  - Model: `sentence-transformers/paraphrase-MiniLM-L6-v2`
  - Use cases: search for patterns, logs, alerts
  - Vectors stored in Redis vector index (HNSW)

---

## 🧪 Simulated Devices

3–5 fake devices using:

- Location: Building A/B/C
- Type: HVAC, Lighting, Server
- Energy values: 0.5 to 10 kWh with occasional spikes
- Generated using `faker + schedule`

---

## 🎨 Frontend: FastAPI Web Interface

Modern responsive web interface featuring:

- **Dashboard**: Real-time energy metrics and system overview
- **Device Management**: Comprehensive device listing and details
- **Analytics**: Interactive charts and trend analysis
- **Search**: Semantic search functionality for device logs and patterns
- **Admin Panel**: Device registration and configuration
- **Mobile-Friendly**: Responsive design that works on all devices

---

## 💡 Future Extensions

- Device health prediction
- Feedback loop to retrain models
- Alerts via email/webhooks
- Integration with Grafana or Supabase

---
