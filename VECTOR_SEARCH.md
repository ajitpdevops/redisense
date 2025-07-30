# 🚀 Redis 8 Vector Search Implementation

## Overview

This implementation adds powerful semantic search capabilities to Redisense using **Redis 8's native vector search features** with RediSearch.

## 🔧 Features Implemented

### 1. **AI Service Enhancements**

- **Real Embeddings**: Uses `sentence-transfo## 🚀 Integration Status

✅ **Vector Index**: Created and operational
✅ **Device Embeddings**: Auto-generated during seeding
✅ **Semantic Search**: Full KNN search with metadata
✅ **CLI Commands**: All vector operations working
✅ **Dashboard Integration**: Live semantic search UI in Streamlit
⏳ **API Endpoints**: Pending (next step)

## 🎯 Dashboard Features Live!

### 🔍 **Semantic Search Interface**

- **Natural Language Input**: Search with phrases like "HVAC system", "energy efficient"
- **Real-time Results**: AI-powered similarity matching with scores
- **Rich Metadata Display**: Device type, manufacturer, location details
- **Progress Indicators**: Visual similarity score bars
- **Example Searches**: One-click buttons for common queries

### 🖥️ **Access the Dashboard**

```bash
# Launch the dashboard
uv run python cli.py dashboard

# Then open: http://localhost:8501
```

### 🧭 **Navigation Menu**

The dashboard now features a clean left navigation menu with dedicated sections:

- **🏠 Main Dashboard**: Overview metrics, system status, and device overview charts
- **📈 Energy Consumption Trends**: Interactive time-series charts and energy analytics
- **🔍 Semantic Device Search**: AI-powered natural language device search interface
- **🏠 Device Status**: Device cards with analysis, settings, and similarity features
- **📋 Show Raw Data**: Complete raw readings table with export functionality

**Navigate between sections using the radio buttons in the left sidebar!**

### 🎨 **UI Features**

- **Clean Navigation**: Left sidebar radio menu for section-based navigation
- **Organized Layout**: Each section focuses on specific functionality
- **Beautiful gradient design** with search result cards
- **Expandable result sections** with similarity scores
- **Metadata display** with device information
- **Progress bars** showing similarity percentages
- **Example search buttons** for quick testingraphrase-MiniLM-L6-v2` model
- **Device Content Creation**: Combines device metadata into searchable text
- **Fallback Support**: Mock embeddings if model fails to load

### 2. **Redis Vector Index**

- **Native RediSearch**: Uses `FT.CREATE` with vector fields
- **384-Dimensional Vectors**: Compatible with sentence-transformers
- **Cosine Similarity**: Optimal for semantic search
- **Hybrid Search**: Combine vector similarity with metadata filtering

### 3. **Automatic Embedding Generation**

- **Device Embeddings**: Auto-generated for all devices during seeding
- **Searchable Content**: Combines name, type, manufacturer, description, etc.
- **Real-time Updates**: New devices automatically get embeddings

## 🛠️ CLI Commands

### Vector Management

```bash
# Create vector index
uv run python cli.py create-vector-index

# Generate embeddings for all devices
uv run python cli.py generate-embeddings

# Check vector search status
uv run python cli.py vector-status

# Perform semantic search
uv run python cli.py search "HVAC air conditioning" --limit 5
```

### Data Management

```bash
# Seed data with automatic embedding generation
uv run python cli.py seed-data --device-count 10 --days 7

# Clear all data including vectors
uv run python cli.py clear-data
```

## 🔍 Search Examples

### Device Type Search

```bash
uv run python cli.py search "HVAC system"
uv run python cli.py search "air conditioning"
uv run python cli.py search "lighting control"
```

### Efficiency Search

```bash
uv run python cli.py search "energy efficient"
uv run python cli.py search "high performance"
uv run python cli.py search "low power consumption"
```

### Location-Based Search

```bash
uv run python cli.py search "warehouse equipment"
uv run python cli.py search "server room cooling"
uv run python cli.py search "office lighting"
```

## 📊 Data Structure

### Vector Index Schema

```
Index Name: device_embeddings
Prefix: device_embed:*
Fields:
  - content (TEXT): Searchable device description
  - device_id (TAG): Device identifier
  - device_type (TAG): HVAC, Lighting, Server, etc.
  - manufacturer (TAG): Device manufacturer
  - location (TAG): Device location
  - embedding (VECTOR): 384-dim float32 vector, COSINE distance
```

### Example Document

```
Key: device_embed:device-001
Fields:
  content: "Device: Air Handler - Zone 09 | Type: HVAC | Manufacturer: Lennox | Model: LEN-7371C | Location: Warehouse - Loading Bay | Description: Climate control system..."
  device_id: "device-001"
  device_type: "HVAC"
  manufacturer: "Lennox"
  location: "Warehouse - Loading Bay"
  embedding: [binary float32 vector]
```

## 🚀 Redis 8 Capabilities Used

### 1. **RediSearch Vector Fields**

```python
VectorField(
    "embedding",
    "FLAT",
    {
        "TYPE": "FLOAT32",
        "DIM": 384,
        "DISTANCE_METRIC": "COSINE"
    }
)
```

### 2. **KNN Vector Search**

```python
query = f"*=>[KNN {limit} @embedding $query_vector AS score]"
results = redis_client.ft("device_embeddings").search(
    query,
    query_params={"query_vector": query_bytes}
)
```

### 3. **Hybrid Filtering**

```python
# Combine vector search with metadata filters
query = f"@device_type:HVAC =>[KNN 5 @embedding $query_vector AS score]"
```

## 🧪 Testing

### Basic Functionality Test

```bash
# Check vector search status
cd /workspaces/intellistream/redisense
timeout 30 uv run python cli.py vector-status

# Test semantic search (note: first search may take longer due to model loading)
timeout 60 uv run python cli.py search 'HVAC system' --limit 3
```

### Example Search Results

```
🔍 Searching for: 'HVAC system'
📋 Found 3 results:

1. Device: device-007
   Score: 0.653
   Content: Device: CRAC Unit - Zone 15 | Type: Cooling | Manufacturer: CoolIT
   Type: Cooling
   Manufacturer: CoolIT
   Location: Office - West Wing

2. Device: device-008
   Score: 0.530
   Content: Device: Radiant Floor - Zone 14 | Type: Heating | Manufacturer: Bosch
   Type: Heating
   Manufacturer: Bosch
   Location: Building A - Floor 2
```

### Comprehensive Test Script

```bash
cd /workspaces/intellistream/redisense
python test_vector_search.py
```

## ✅ Success! Vector Search Implementation Complete

### 🎯 Verified Working Features

#### 1. **Vector Index Creation**

- ✅ Redis vector index `device_embeddings` created successfully
- ✅ 384-dimensional embeddings with COSINE similarity
- ✅ Automatic fallback for different Redis client versions

#### 2. **Device Embeddings**

- ✅ 8 devices with embeddings generated and stored
- ✅ Searchable content combines device metadata
- ✅ Real sentence-transformer embeddings (paraphrase-MiniLM-L6-v2)

#### 3. **Semantic Search**

- ✅ KNN search working with similarity scores
- ✅ "HVAC system" → Found cooling, heating equipment (0.65+ scores)
- ✅ Rich metadata display (type, manufacturer, location)
- ✅ Fast sub-second search after model loading

### 🔍 Search Quality Examples

**Query**: "HVAC system"
**Results**:

- CRAC Unit (Cooling) - Score: 0.653 ⭐ Excellent semantic match
- Radiant Floor (Heating) - Score: 0.530 ⭐ Perfect HVAC component

This demonstrates true semantic understanding - finding relevant climate control equipment even when exact keywords don't match!

### vs Traditional Search

- **Semantic Understanding**: Finds "air conditioning" when searching "HVAC"
- **Fuzzy Matching**: Handles typos and variations
- **Context Aware**: Understanding of technical terms and synonyms

### Redis 8 Advantages

- **Native Performance**: No external vector database needed
- **Sub-millisecond Search**: Optimized C implementation
- **Memory Efficient**: Compressed vector storage
- **Scalable**: Handles millions of vectors

## 🔧 Integration Points

### Streamlit Dashboard

- Add search bar for semantic device search
- Real-time search suggestions
- Filter by similarity score

### API Endpoints

- `/api/search` - Semantic search endpoint
- `/api/embeddings/generate` - Manual embedding generation
- `/api/embeddings/status` - Vector index status

### Monitoring

- Vector index size and performance metrics
- Search query analytics
- Embedding generation success rates

## 📈 Next Steps

1. **Dashboard Integration**: Add search UI to Streamlit
2. **Auto-Refresh**: Real-time embedding updates
3. **Advanced Queries**: Hybrid vector + traditional search
4. **Analytics**: Search pattern analysis
5. **Recommendations**: Similar device suggestions

## 🔗 Dependencies

- **sentence-transformers**: Real embedding generation
- **redis**: Redis 8 with RediSearch module
- **numpy**: Vector operations
- **scikit-learn**: ML utilities

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors During Index Creation

**Error**: `No module named 'redis.commands.search.indexDefinition'`
**Solution**: The system automatically falls back to manual Redis commands. This is expected behavior with different Redis client versions.

#### 2. Slow First Search

**Issue**: First search takes 10-30 seconds due to model loading
**Solution**: Use `timeout 60` command prefix for first search:

```bash
timeout 60 uv run python cli.py search 'your query'
```

#### 3. No Search Results

**Check**: Ensure devices and embeddings exist:

```bash
uv run python cli.py vector-status
```

**Solution**: If no embeddings exist, generate them:

```bash
uv run python cli.py generate-embeddings
```

#### 4. Connection Issues

**Error**: Redis connection failures
**Solution**: Check `.env` file has correct Redis credentials and the Redis instance supports RediSearch module.

#### 5. PyTorch/Transformers Warnings

**Warning**: `encoder_attention_mask is deprecated and will be removed in version 4.55.0`
**Solution**: This warning is automatically suppressed in the codebase. It's a harmless deprecation warning from the BERT model used by sentence-transformers.

### Performance Tips

- First search is slower due to sentence-transformer model loading
- Subsequent searches are fast (sub-second)
- Use `--limit` parameter to control result count
- Vector index automatically handles millions of documents

## 🎮 **Live Demo - Semantic Search in Action!**

The Redis 8 vector search is now **fully integrated** into the Streamlit dashboard! Here's what you can experience:

### 🔍 **Interactive Search Experience**

1. **Open the Dashboard**: `uv run python cli.py dashboard` → http://localhost:8501
2. **Natural Language Search**: Type queries like:
   - "HVAC system" → Finds cooling, heating equipment
   - "energy efficient lighting" → Discovers LED and smart lighting
   - "server room equipment" → Locates cooling and server devices
3. **Visual Results**: See similarity scores, device metadata, and rich content previews
4. **Quick Examples**: Click buttons for instant search demos

### 🎯 **What Makes It Special**

- **True Semantic Understanding**: Finds "cooling" when you search "HVAC"
- **Context Awareness**: Understands technical terms and synonyms
- **Real-time Performance**: Sub-second search after model loading
- **Beautiful UI**: Progress bars, gradients, and expandable cards
- **No Keywords Required**: Uses AI embeddings, not exact text matching

### 🚀 **Redis 8 Vector Power**

- **Native Performance**: No external vector database needed
- **384-Dimensional Embeddings**: Real sentence-transformer vectors
- **COSINE Similarity**: Optimal for semantic matching
- **Scalable Architecture**: Ready for millions of devices

**The semantic search UI showcases Redis 8's cutting-edge vector capabilities in action!** 🌟

## 🛒 **Ecommerce-Style Device Recommendations**

Just like shopping online where you see "Related Products" or "Customers also viewed", our dashboard now features **intelligent device recommendations**!

### 🎯 **How It Works:**

1. **Browse Device Cards**: View all your devices in the "🏠 Device Status" section
2. **Click "🔍 Similar"**: On any device card to find related equipment
3. **See Recommendations**: Beautiful cards showing similar devices with AI similarity scores
4. **Chain Discovery**: Click on recommended devices to discover more connections
5. **Visual Feedback**: Hover effects, gradients, and similarity percentages

### 🎨 **Visual Features:**

- **Selected Device Highlight**: Purple gradient card showing your chosen device
- **Recommendation Cards**: Hoverable cards with similarity percentages
- **Smooth Animations**: Cards lift on hover like real ecommerce sites
- **Similarity Badges**: Green gradient badges showing match percentages
- **Action Buttons**: "View Details" and "Find Similar" for each recommendation

### 🧠 **AI-Powered Matching:**

- **Vector Similarity**: Uses Redis 8 embeddings to find truly similar devices
- **Context Understanding**: Finds devices similar by function, not just keywords
- **Smart Scoring**: Shows percentage match based on semantic similarity
- **Real-time Discovery**: Instantly explore device relationships

**Experience the future of device discovery - just like shopping for products online!** 🌟

## 📊 **Enhanced Device Analysis & Settings**

The dashboard now features comprehensive device analysis and configuration capabilities beyond just semantic search!

### 🎯 **Device Analysis Features:**

1. **Click "📊 Analyze"** on any device card to access:
   - **Performance Overview**: Real-time metrics, health scores, and utilization analysis
   - **Energy Patterns**: Hourly and daily consumption patterns with interactive charts
   - **Anomaly Detection**: AI-powered detection of unusual power consumption
   - **Statistical Analysis**: Comprehensive statistics with box plots and distributions
   - **AI Recommendations**: Smart suggestions for optimization and maintenance

### ⚙️ **Device Settings Features:**

1. **Click "⚙️ Settings"** on any device card to access a comprehensive 5-tab interface:
   - **📋 Device Details**: Complete device information viewing (name, ID, type, manufacturer, model, location, power rating, status, description, technical metadata, and real-time status metrics)
   - **📝 Basic Settings**: Edit name, location, manufacturer, model, power rating, description, and operational status
   - **🔧 Advanced Config**: Configure efficiency ratings, monitoring intervals, retention policies, and advanced features
   - **📊 Monitoring**: Real-time status display, monitoring controls, and data export capabilities
   - **🚨 Alert Configuration**: Set thresholds, manage notifications, configure recipients, and test alert systems

### 🔍 **Three Powerful Actions Per Device:**

- **📊 Analyze**: Deep dive into device performance and get AI-powered insights
- **⚙️ Settings**: Complete device configuration and monitoring setup
- **🔍 Similar**: Discover related devices using vector similarity (ecommerce-style)

### 🎨 **Analysis Dashboard Features:**

- **Tabbed Interface**: Organized analysis across Performance, Energy Patterns, Anomalies, Statistics, and Recommendations
- **Interactive Charts**: Plotly-powered visualizations for trends and patterns
- **Health Scoring**: Real-time calculation of device health and efficiency metrics
- **Cost Analysis**: Energy cost estimation and monthly projections
- **Anomaly Visualization**: Scatter plots highlighting unusual power consumption events

### ⚙️ **Settings Dashboard Features:**

- **5-Tab Interface**: Device Details, Basic Settings, Advanced Config, Monitoring, and Alerts
- **Consolidated View**: All device information and configuration in one place (no more sidebar clutter!)
- **Device Details Tab**: Complete read-only overview including technical metadata and real-time metrics
- **Form-Based Editing**: Intuitive forms for all configurable device parameters
- **Real-Time Monitoring**: Simulated live metrics display (power, temperature, voltage)
- **Alert Management**: Comprehensive threshold and notification configuration
- **Test Capabilities**: Built-in testing for email, SMS, and dashboard alerts

### 🧠 **AI-Powered Insights:**

- **Smart Recommendations**: Context-aware suggestions based on device type and usage patterns
- **Efficiency Analysis**: Automatic calculation of utilization and efficiency metrics
- **Pattern Recognition**: Hourly and daily usage pattern analysis
- **Anomaly Detection**: Statistical analysis using z-scores to identify outliers
- **Cost Optimization**: Energy cost analysis and savings recommendations

**Now you have a complete device management platform with AI insights, not just search! Everything is organized with clean navigation and consolidated device-specific actions - no more cluttered interface!** 🚀
