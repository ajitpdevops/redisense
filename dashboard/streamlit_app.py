import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
from datetime import datetime, timedelta
import numpy as np
import sys
import os

# Suppress PyTorch/Transformers deprecation warnings
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings
from app.services.redis_service import RedisService
from app.services.ai_service import AIService
from app.services.ai_service import AIService

# Page configuration
st.set_page_config(
    page_title="Redisense - Energy Monitoring Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .device-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .search-result {
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    .similarity-score {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .search-header {
        background: linear-gradient(90deg, #6f42c1 0%, #e83e8c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .recommendation-card {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
        border-color: #007bff;
    }
    .similarity-badge {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
        display: inline-block;
        margin-top: 8px;
    }
    .device-title {
        margin: 0 0 12px 0;
        color: #2c3e50;
        font-size: 1.1em;
        font-weight: bold;
    }
    .device-info {
        margin: 4px 0;
        color: #495057;
        font-size: 0.9em;
    }
    .selected-device-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    .status-normal { color: #28a745; }
    .status-maintenance { color: #ffc107; }
    .status-anomaly { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=180)  # Cache for 3 minutes
def load_data():
    """Load data from Redis with caching"""
    try:
        settings = Settings()
        redis_service = RedisService(settings)

        # Get all devices
        devices = redis_service.get_all_devices()

        # Get recent readings for all devices
        all_readings = []
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)

        for device in devices:
            readings = redis_service.get_energy_readings(
                device.device_id,
                start_time,
                end_time,
                limit=200
            )
            all_readings.extend(readings)

        return devices, all_readings, True, None
    except Exception as e:
        return [], [], False, str(e)

def create_device_overview_chart(devices):
    """Create device overview chart"""
    if not devices:
        return None

    device_types = {}
    device_statuses = {}

    for device in devices:
        device_type = device.device_type.value if hasattr(device.device_type, 'value') else str(device.device_type)
        device_status = device.status.value if hasattr(device.status, 'value') else str(device.status)

        device_types[device_type] = device_types.get(device_type, 0) + 1
        device_statuses[device_status] = device_statuses.get(device_status, 0) + 1

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=["Device Types", "Device Status"]
    )

    # Device types pie chart
    fig.add_trace(
        go.Pie(
            labels=list(device_types.keys()),
            values=list(device_types.values()),
            name="Device Types",
            hole=0.4
        ),
        row=1, col=1
    )

    # Device status pie chart
    colors = {'normal': '#28a745', 'maintenance': '#ffc107', 'anomaly': '#dc3545'}
    status_colors = [colors.get(status, '#6c757d') for status in device_statuses.keys()]

    fig.add_trace(
        go.Pie(
            labels=list(device_statuses.keys()),
            values=list(device_statuses.values()),
            name="Device Status",
            hole=0.4,
            marker_colors=status_colors
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="Device Overview",
        height=400,
        showlegend=True
    )

    return fig

def create_energy_trends_chart(readings, devices=None, time_range_hours=None):
    """Create energy consumption trends chart with device names and time filtering"""
    if not readings:
        return None

    # Filter readings by time range if specified
    if time_range_hours is not None:
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        readings = [r for r in readings if r.timestamp >= cutoff_time]

        if not readings:
            return None

    # Create device name mapping
    device_names = {}
    if devices:
        for device in devices:
            device_names[device.device_id] = getattr(device, 'name', device.device_id)

    # Convert to DataFrame
    df_data = []
    for reading in readings:
        device_display_name = device_names.get(reading.device_id, reading.device_id)
        df_data.append({
            'device_id': reading.device_id,
            'device_name': device_display_name,
            'timestamp': reading.timestamp,
            'energy_kwh': reading.energy_kwh,
            'power_kw': reading.power_kw if reading.power_kw else reading.energy_kwh,
            'voltage': reading.voltage if reading.voltage else None,
            'current': reading.current if reading.current else None
        })

    df = pd.DataFrame(df_data)

    if df.empty:
        return None

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Create line chart with device names
    time_range_text = ""
    if time_range_hours is not None:
        if time_range_hours < 1:
            minutes = int(time_range_hours * 60)
            time_range_text = f" (Last {minutes} Minutes)"
        else:
            hours = int(time_range_hours) if time_range_hours == int(time_range_hours) else time_range_hours
            time_range_text = f" (Last {hours} Hour{'s' if hours != 1 else ''})"

    fig = px.line(
        df,
        x='timestamp',
        y='energy_kwh',
        color='device_name',  # Use device names instead of IDs
        title=f'Energy Consumption Over Time{time_range_text}',
        labels={
            'energy_kwh': 'Energy (kWh)',
            'timestamp': 'Time',
            'device_name': 'Device'
        }
    )

    fig.update_layout(
        height=500,
        xaxis_title="Time",
        yaxis_title="Energy Consumption (kWh)",
        hovermode='x unified'
    )

    return fig

def create_real_time_metrics(readings):
    """Create real-time metrics"""
    if not readings:
        return {}

    # Get latest readings per device
    latest_readings = {}
    for reading in readings:
        if reading.device_id not in latest_readings:
            latest_readings[reading.device_id] = reading
        elif reading.timestamp > latest_readings[reading.device_id].timestamp:
            latest_readings[reading.device_id] = reading

    # Calculate metrics
    current_power = sum(r.power_kw or r.energy_kwh for r in latest_readings.values())
    total_devices = len(latest_readings)
    avg_consumption = current_power / total_devices if total_devices > 0 else 0

    # Calculate total energy over last 24 hours
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)
    total_energy_24h = sum(
        r.energy_kwh for r in readings
        if r.timestamp >= last_24h
    )

    return {
        'current_power': current_power,
        'total_devices': total_devices,
        'avg_consumption': avg_consumption,
        'total_energy_24h': total_energy_24h,
        'latest_readings': latest_readings
    }

def create_device_cards(devices, metrics):
    """Create device status cards with proper names and profile views"""
    if not devices:
        return

    latest_readings = metrics.get('latest_readings', {})

    # Create device cards grid
    st.subheader("🏠 Device Status Overview")

    # Create columns for device cards (4 per row)
    num_cols = 4
    for i in range(0, len(devices), num_cols):
        cols = st.columns(num_cols)
        for j, device in enumerate(devices[i:i+num_cols]):
            with cols[j]:
                latest_reading = latest_readings.get(device.device_id)

                device_type = device.device_type.value if hasattr(device.device_type, 'value') else str(device.device_type)
                device_status = device.status.value if hasattr(device.status, 'value') else str(device.status)

                status_class = f"status-{device_status}"
                status_color = {"normal": "🟢", "maintenance": "🟡", "anomaly": "🔴"}.get(device_status, "⚪")

                current_power = latest_reading.power_kw if latest_reading and latest_reading.power_kw else 0
                last_update = latest_reading.timestamp.strftime("%H:%M:%S") if latest_reading else "No data"

                # Create expandable device card
                device_name = getattr(device, 'name', device.device_id)
                with st.expander(f"{status_color} {device_name}", expanded=False):
                    st.markdown(f"**ID:** {device.device_id}")
                    st.markdown(f"**Type:** {device_type}")
                    st.markdown(f"**Manufacturer:** {getattr(device, 'manufacturer', 'Unknown')}")
                    st.markdown(f"**Model:** {getattr(device, 'model', 'Unknown')}")
                    st.markdown(f"**Location:** {getattr(device, 'location', 'Unknown')}")
                    power_rating = getattr(device, 'power_rating', None)
                    st.markdown(f"**Power Rating:** {power_rating}W" if power_rating else "**Power Rating:** Unknown")
                    st.markdown(f"**Status:** <span class='{status_class}'>{device_status.title()}</span>", unsafe_allow_html=True)

                    description = getattr(device, 'description', None)
                    if description:
                        st.markdown(f"**Description:** {description}")

                    st.markdown("---")
                    st.markdown(f"**Current Power:** {current_power:.2f} kW")
                    st.markdown(f"**Last Update:** {last_update}")

                    # Add quick actions
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"📊 Analyze", key=f"analyze_{device.device_id}"):
                            st.session_state.device_to_analyze = device.device_id
                    with col2:
                        if st.button(f"⚙️ Settings", key=f"settings_{device.device_id}"):
                            st.session_state.device_to_configure = device.device_id
                    with col3:
                        if st.button(f"🔍 Similar", key=f"similar_{device.device_id}"):
                            # Store the selected device for similarity search
                            st.session_state.selected_device_for_similarity = device.device_id

def find_similar_devices(selected_device_id, devices, ai_service, limit=4):
    """Find devices similar to the selected device using vector embeddings"""
    try:
        # Find the selected device
        selected_device = next((d for d in devices if d.device_id == selected_device_id), None)
        if not selected_device:
            return []

        # Create search content from the selected device
        device_content = f"Device: {getattr(selected_device, 'name', selected_device.device_id)} | Type: {selected_device.device_type} | Manufacturer: {getattr(selected_device, 'manufacturer', '')} | Location: {getattr(selected_device, 'location', '')} | Description: {getattr(selected_device, 'description', '')}"

        # Perform semantic search
        similar_results = ai_service.semantic_search(device_content, limit + 1)  # +1 to exclude self

        # Filter out the selected device itself and return top matches
        similar_devices = []
        for result in similar_results:
            if result['device_id'] != selected_device_id:
                device = next((d for d in devices if d.device_id == result['device_id']), None)
                if device:
                    similar_devices.append({
                        'device': device,
                        'score': result['score'],
                        'content': result['content']
                    })

        return similar_devices[:limit]

    except Exception as e:
        st.error(f"Error finding similar devices: {e}")
        return []

def display_similar_devices_section(devices):
    """Display similar devices section like ecommerce recommendations"""
    if 'selected_device_for_similarity' not in st.session_state:
        return

    selected_device_id = st.session_state.selected_device_for_similarity
    selected_device = next((d for d in devices if d.device_id == selected_device_id), None)

    if not selected_device:
        return

    st.markdown("---")
    st.markdown("## 🔗 Similar Devices")
    st.markdown(f"*Devices similar to **{getattr(selected_device, 'name', selected_device_id)}** - Powered by AI Vector Search*")

    try:
        ai_service = AIService()
        similar_devices = find_similar_devices(selected_device_id, devices, ai_service, limit=3)

        if similar_devices:
            device_type = selected_device.device_type.value if hasattr(selected_device.device_type, 'value') else str(selected_device.device_type)

            # Display selected device info
            with st.container():
                st.markdown(f"""
                <div class="selected-device-highlight">
                    <h3 style="margin: 0 0 15px 0;">📌 Selected Device: {getattr(selected_device, 'name', selected_device_id)}</h3>
                    <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 20px;">
                        <div>
                            <p style="margin: 5px 0;"><strong>Type:</strong> {device_type}</p>
                            <p style="margin: 5px 0;"><strong>Manufacturer:</strong> {getattr(selected_device, 'manufacturer', 'Unknown')}</p>
                            <p style="margin: 5px 0;"><strong>Location:</strong> {getattr(selected_device, 'location', 'Unknown')}</p>
                        </div>
                        <div>
                            <p style="margin: 5px 0;"><strong>Description:</strong> {getattr(selected_device, 'description', 'Device for energy monitoring and control')}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 🎯 Recommended Similar Devices:")

            # Display similar devices in columns like ecommerce
            cols = st.columns(len(similar_devices))

            for i, similar in enumerate(similar_devices):
                with cols[i]:
                    device = similar['device']
                    score = similar['score']

                    device_name = getattr(device, 'name', device.device_id)
                    device_type = device.device_type.value if hasattr(device.device_type, 'value') else str(device.device_type)

                    # Create attractive recommendation card
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4 class="device-title">🔗 {device_name}</h4>
                            <p class="device-info"><strong>Type:</strong> {device_type}</p>
                            <p class="device-info"><strong>Manufacturer:</strong> {getattr(device, 'manufacturer', 'Unknown')}</p>
                            <p class="device-info"><strong>Location:</strong> {getattr(device, 'location', 'Unknown')}</p>
                            <span class="similarity-badge">🎯 {score:.1%} Match</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # Action buttons for similar devices
                        if st.button(f"📊 View Details", key=f"view_similar_{device.device_id}"):
                            st.session_state.selected_device_for_similarity = device.device_id
                            st.rerun()

                        if st.button(f"🔍 Find Similar", key=f"find_similar_{device.device_id}"):
                            st.session_state.selected_device_for_similarity = device.device_id
                            st.rerun()

            # Clear selection button
            if st.button("❌ Clear Selection", key="clear_similarity"):
                if 'selected_device_for_similarity' in st.session_state:
                    del st.session_state.selected_device_for_similarity
                st.rerun()

        else:
            st.info("🔍 No similar devices found. Try generating embeddings: `uv run python cli.py generate-embeddings`")

    except Exception as e:
        st.error(f"❌ Error finding similar devices: {e}")
        st.info("💡 Make sure vector embeddings are generated: `uv run python cli.py generate-embeddings`")

def display_device_analysis(device, readings, devices, metrics):
    """Display comprehensive device analysis"""
    if 'device_to_analyze' not in st.session_state:
        return

    device_id = st.session_state.device_to_analyze
    target_device = next((d for d in devices if d.device_id == device_id), None)

    if not target_device:
        return

    device_name = getattr(target_device, 'name', device_id)

    # Create analysis modal
    st.markdown("---")
    st.markdown(f"## 📊 Device Analysis: {device_name}")
    st.markdown(f"*Comprehensive analysis for device: **{device_id}***")

    # Filter readings for this device
    device_readings = [r for r in readings if r.device_id == device_id]

    if not device_readings:
        st.warning("⚠️ No readings available for analysis")
        if st.button("❌ Close Analysis", key="close_analysis"):
            del st.session_state.device_to_analyze
            st.rerun()
        return

    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Performance", "⚡ Energy Patterns", "🔍 Anomalies", "📊 Statistics", "🎯 Recommendations"])

    with tab1:
        st.subheader("🎯 Performance Overview")

        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)

        latest_reading = max(device_readings, key=lambda x: x.timestamp) if device_readings else None
        avg_power = sum(r.power_kw or r.energy_kwh for r in device_readings) / len(device_readings)
        max_power = max(r.power_kw or r.energy_kwh for r in device_readings)
        min_power = min(r.power_kw or r.energy_kwh for r in device_readings)

        with col1:
            current_power = latest_reading.power_kw if latest_reading and latest_reading.power_kw else 0
            st.metric("Current Power", f"{current_power:.2f} kW",
                     delta=f"{current_power - avg_power:.2f} kW" if avg_power else None)

        with col2:
            st.metric("Average Power", f"{avg_power:.2f} kW")

        with col3:
            st.metric("Peak Power", f"{max_power:.2f} kW")

        with col4:
            st.metric("Min Power", f"{min_power:.2f} kW")

        # Power trend for this device
        if len(device_readings) > 1:
            df_device = pd.DataFrame([{
                'timestamp': r.timestamp,
                'power_kw': r.power_kw or r.energy_kwh,
                'energy_kwh': r.energy_kwh
            } for r in device_readings])

            fig_power = px.line(df_device, x='timestamp', y='power_kw',
                              title=f"Power Consumption - {device_name}")
            st.plotly_chart(fig_power, use_container_width=True)

        # Device health score
        power_rating = getattr(target_device, 'power_rating', None)
        if power_rating:
            utilization = (current_power * 1000) / power_rating * 100  # Convert kW to W
            st.subheader("🏥 Device Health")

            health_col1, health_col2 = st.columns(2)
            with health_col1:
                st.metric("Power Utilization", f"{utilization:.1f}%")

                if utilization > 90:
                    st.error("⚠️ High utilization - Consider load balancing")
                elif utilization > 70:
                    st.warning("⚠️ Medium utilization - Monitor closely")
                else:
                    st.success("✅ Normal utilization")

            with health_col2:
                efficiency = (min_power / max_power * 100) if max_power > 0 else 0
                st.metric("Power Efficiency", f"{efficiency:.1f}%")

                if efficiency > 80:
                    st.success("✅ Excellent efficiency")
                elif efficiency > 60:
                    st.info("ℹ️ Good efficiency")
                else:
                    st.warning("⚠️ Low efficiency - Check for issues")

    with tab2:
        st.subheader("⚡ Energy Consumption Patterns")

        if len(device_readings) > 1:
            # Time-based analysis
            df_device = pd.DataFrame([{
                'timestamp': r.timestamp,
                'hour': r.timestamp.hour,
                'day_of_week': r.timestamp.weekday(),
                'energy_kwh': r.energy_kwh,
                'power_kw': r.power_kw or r.energy_kwh
            } for r in device_readings])

            # Hourly pattern
            hourly_avg = df_device.groupby('hour')['power_kw'].mean()
            fig_hourly = px.bar(x=hourly_avg.index, y=hourly_avg.values,
                              title="Average Power by Hour of Day",
                              labels={'x': 'Hour', 'y': 'Average Power (kW)'})
            st.plotly_chart(fig_hourly, use_container_width=True)

            # Daily pattern
            daily_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily_avg = df_device.groupby('day_of_week')['power_kw'].mean()
            fig_daily = px.bar(x=[daily_names[i] for i in daily_avg.index], y=daily_avg.values,
                             title="Average Power by Day of Week",
                             labels={'x': 'Day', 'y': 'Average Power (kW)'})
            st.plotly_chart(fig_daily, use_container_width=True)

            # Energy distribution
            st.subheader("📊 Energy Distribution")
            fig_hist = px.histogram(df_device, x='power_kw', nbins=20,
                                  title="Power Consumption Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("🔍 Anomaly Detection")

        if len(device_readings) > 10:
            powers = [r.power_kw or r.energy_kwh for r in device_readings]
            mean_power = np.mean(powers)
            std_power = np.std(powers)

            # Simple anomaly detection using standard deviation
            anomalies = []
            for i, reading in enumerate(device_readings):
                power = reading.power_kw or reading.energy_kwh
                z_score = abs((power - mean_power) / std_power) if std_power > 0 else 0

                if z_score > 2:  # 2 standard deviations
                    anomalies.append({
                        'timestamp': reading.timestamp,
                        'power': power,
                        'z_score': z_score,
                        'severity': 'High' if z_score > 3 else 'Medium'
                    })

            if anomalies:
                st.warning(f"⚠️ Found {len(anomalies)} potential anomalies")

                for anomaly in anomalies[-5:]:  # Show last 5 anomalies
                    severity_color = "🔴" if anomaly['severity'] == 'High' else "🟡"
                    st.write(f"{severity_color} **{anomaly['severity']}** anomaly at {anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"   Power: {anomaly['power']:.2f} kW (Z-score: {anomaly['z_score']:.2f})")

                # Anomaly visualization
                df_anomaly = pd.DataFrame([{
                    'timestamp': r.timestamp,
                    'power_kw': r.power_kw or r.energy_kwh,
                    'is_anomaly': any(a['timestamp'] == r.timestamp for a in anomalies)
                } for r in device_readings])

                fig_anomaly = px.scatter(df_anomaly, x='timestamp', y='power_kw',
                                       color='is_anomaly',
                                       title="Power Consumption with Anomalies",
                                       color_discrete_map={True: 'red', False: 'blue'})
                st.plotly_chart(fig_anomaly, use_container_width=True)
            else:
                st.success("✅ No significant anomalies detected")
        else:
            st.info("📊 Need more data points for anomaly detection (minimum 10 readings)")

    with tab4:
        st.subheader("📊 Statistical Analysis")

        powers = [r.power_kw or r.energy_kwh for r in device_readings]
        energies = [r.energy_kwh for r in device_readings]

        # Statistical metrics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Power Statistics (kW)**")
            st.metric("Mean", f"{np.mean(powers):.3f}")
            st.metric("Median", f"{np.median(powers):.3f}")
            st.metric("Std Dev", f"{np.std(powers):.3f}")
            st.metric("Min", f"{np.min(powers):.3f}")
            st.metric("Max", f"{np.max(powers):.3f}")

        with col2:
            st.markdown("**Energy Statistics (kWh)**")
            st.metric("Total Energy", f"{np.sum(energies):.3f}")
            st.metric("Avg per Reading", f"{np.mean(energies):.3f}")
            st.metric("Peak Reading", f"{np.max(energies):.3f}")

            # Calculate uptime
            time_span = max(r.timestamp for r in device_readings) - min(r.timestamp for r in device_readings)
            hours_span = time_span.total_seconds() / 3600
            st.metric("Monitoring Period", f"{hours_span:.1f} hours")

        # Box plot
        fig_box = px.box(y=powers, title="Power Consumption Distribution")
        st.plotly_chart(fig_box, use_container_width=True)

    with tab5:
        st.subheader("🎯 AI-Powered Recommendations")

        # Calculate recommendation factors
        powers = [r.power_kw or r.energy_kwh for r in device_readings]
        avg_power = np.mean(powers)
        power_rating = getattr(target_device, 'power_rating', None)
        device_type = target_device.device_type.value if hasattr(target_device.device_type, 'value') else str(target_device.device_type)

        recommendations = []

        # Power efficiency recommendations
        if power_rating:
            utilization = (avg_power * 1000) / power_rating * 100
            if utilization > 85:
                recommendations.append({
                    'type': '⚡ Power Management',
                    'priority': 'High',
                    'title': 'High Power Utilization Detected',
                    'description': f'Device is running at {utilization:.1f}% capacity. Consider load balancing or upgrading.',
                    'action': 'Schedule maintenance review and consider capacity planning.'
                })
            elif utilization < 30:
                recommendations.append({
                    'type': '💡 Efficiency',
                    'priority': 'Medium',
                    'title': 'Low Power Utilization',
                    'description': f'Device is only using {utilization:.1f}% capacity. This may indicate oversizing.',
                    'action': 'Consider right-sizing or consolidating workloads.'
                })

        # Variance-based recommendations
        power_variance = np.var(powers)
        if power_variance > (avg_power * 0.3) ** 2:  # High variance
            recommendations.append({
                'type': '📊 Stability',
                'priority': 'Medium',
                'title': 'High Power Variance Detected',
                'description': 'Power consumption shows significant variation, which may indicate inefficient operation.',
                'action': 'Monitor for operational patterns and consider load stabilization.'
            })

        # Device-specific recommendations
        if 'hvac' in device_type.lower() or 'air' in device_type.lower():
            recommendations.append({
                'type': '🌡️ HVAC Optimization',
                'priority': 'Low',
                'title': 'HVAC System Optimization',
                'description': 'Consider implementing smart scheduling and temperature setback strategies.',
                'action': 'Install programmable thermostats and zone control systems.'
            })
        elif 'server' in device_type.lower() or 'computer' in device_type.lower():
            recommendations.append({
                'type': '🖥️ IT Equipment',
                'priority': 'Low',
                'title': 'Server Power Management',
                'description': 'Enable power management features and consider virtualization.',
                'action': 'Implement server consolidation and power-efficient hardware.'
            })
        elif 'lighting' in device_type.lower():
            recommendations.append({
                'type': '💡 Lighting Efficiency',
                'priority': 'Low',
                'title': 'Lighting Optimization',
                'description': 'Consider LED upgrades and motion sensor integration.',
                'action': 'Implement smart lighting controls and occupancy sensors.'
            })

        # Time-based recommendations
        if len(device_readings) > 24:  # If we have enough data
            df_device = pd.DataFrame([{
                'hour': r.timestamp.hour,
                'power_kw': r.power_kw or r.energy_kwh
            } for r in device_readings])

            hourly_avg = df_device.groupby('hour')['power_kw'].mean()
            peak_hour = hourly_avg.idxmax()
            off_peak_hour = hourly_avg.idxmin()

            recommendations.append({
                'type': '⏰ Scheduling',
                'priority': 'Medium',
                'title': 'Time-Based Optimization',
                'description': f'Peak usage at {peak_hour}:00, lowest at {off_peak_hour}:00.',
                'action': f'Consider scheduling non-critical operations during off-peak hours ({off_peak_hour}:00).'
            })

        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations):
                priority_color = {
                    'High': '🔴',
                    'Medium': '🟡',
                    'Low': '🟢'
                }.get(rec['priority'], '⚪')

                with st.expander(f"{priority_color} {rec['type']}: {rec['title']}", expanded=rec['priority'] == 'High'):
                    st.markdown(f"**Priority:** {rec['priority']}")
                    st.markdown(f"**Description:** {rec['description']}")
                    st.markdown(f"**Recommended Action:** {rec['action']}")
        else:
            st.success("✅ No specific recommendations at this time. Device is operating within normal parameters.")

        # Energy cost estimation
        st.subheader("💰 Cost Analysis")

        total_energy = sum(energies)
        cost_per_kwh = 0.12  # Example rate: $0.12 per kWh
        estimated_cost = total_energy * cost_per_kwh

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Energy Consumed", f"{total_energy:.2f} kWh")
        with col2:
            st.metric("Estimated Cost", f"${estimated_cost:.2f}")
        with col3:
            daily_avg = total_energy / max(1, hours_span / 24) if 'hours_span' in locals() else 0
            monthly_estimate = daily_avg * 30 * cost_per_kwh
            st.metric("Monthly Estimate", f"${monthly_estimate:.2f}")

    # Close button
    if st.button("❌ Close Analysis", key="close_analysis"):
        del st.session_state.device_to_analyze
        st.rerun()

def display_device_settings(device, devices, redis_service):
    """Display comprehensive device settings and configuration"""
    if 'device_to_configure' not in st.session_state:
        return

    device_id = st.session_state.device_to_configure
    target_device = next((d for d in devices if d.device_id == device_id), None)

    if not target_device:
        return

    device_name = getattr(target_device, 'name', device_id)

    # Create settings modal
    st.markdown("---")
    st.markdown(f"## ⚙️ Device Settings: {device_name}")
    st.markdown(f"*Configuration and management for device: **{device_id}***")

    # Settings tabs - now including Device Details as the first tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["� Device Details", "�📝 Basic Settings", "🔧 Advanced Config", "📊 Monitoring", "🚨 Alerts"])

    with tab1:
        st.subheader("📋 Device Information")

        # Display comprehensive device details (previously in sidebar)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📍 Basic Information**")
            st.markdown(f"**Name:** {getattr(target_device, 'name', device_id)}")
            st.markdown(f"**Device ID:** {device_id}")
            st.markdown(f"**Type:** {target_device.device_type.value if hasattr(target_device.device_type, 'value') else str(target_device.device_type)}")
            st.markdown(f"**Manufacturer:** {getattr(target_device, 'manufacturer', 'Unknown')}")
            st.markdown(f"**Model:** {getattr(target_device, 'model', 'Unknown')}")
            st.markdown(f"**Location:** {getattr(target_device, 'location', 'Unknown')}")

            power_rating = getattr(target_device, 'power_rating', None)
            st.markdown(f"**Power Rating:** {power_rating}W" if power_rating else "**Power Rating:** Unknown")
            st.markdown(f"**Status:** {target_device.status.value.title() if hasattr(target_device.status, 'value') else str(target_device.status)}")

        with col2:
            st.markdown("**📝 Description & Details**")
            description = getattr(target_device, 'description', None)
            if description:
                st.markdown(f"**Description:** {description}")
            else:
                st.markdown("**Description:** No description available")

            # Show metadata if available
            metadata = getattr(target_device, 'metadata', {})
            if metadata:
                st.markdown("**🔧 Technical Information**")
                if 'energy_efficiency_rating' in metadata:
                    st.markdown(f"**Efficiency Rating:** {metadata['energy_efficiency_rating']}")
                if 'maintenance_schedule' in metadata:
                    st.markdown(f"**Maintenance Schedule:** {metadata['maintenance_schedule']}")
                if 'operating_hours' in metadata:
                    st.markdown(f"**Operating Hours:** {metadata['operating_hours']:,}")
                if 'last_maintenance' in metadata:
                    st.markdown(f"**Last Service:** {metadata['last_maintenance']}")
                if 'monitoring_interval_seconds' in metadata:
                    st.markdown(f"**Monitoring Interval:** {metadata['monitoring_interval_seconds']}s")
                if 'power_threshold_kw' in metadata:
                    st.markdown(f"**Power Threshold:** {metadata['power_threshold_kw']} kW")
            else:
                st.markdown("**🔧 Technical Information**")
                st.markdown("No technical metadata available")

        # Current readings section
        st.markdown("---")
        st.markdown("**⚡ Current Status**")

        # You can add real-time readings here if available
        import random
        current_time = datetime.now().strftime("%H:%M:%S")
        simulated_power = random.uniform(0.5, 2.0)
        simulated_temp = random.uniform(20, 35)
        simulated_voltage = random.uniform(220, 240)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Power", f"{simulated_power:.2f} kW")

        with col2:
            st.metric("Temperature", f"{simulated_temp:.1f}°C")

        with col3:
            st.metric("Voltage", f"{simulated_voltage:.1f}V")

        with col4:
            st.metric("Last Update", current_time)

    with tab2:
        st.subheader("📝 Basic Device Settings")

        with st.form(f"basic_settings_{device_id}"):
            col1, col2 = st.columns(2)

            with col1:
                # Basic information
                new_name = st.text_input("Device Name",
                                        value=getattr(target_device, 'name', device_id))
                new_location = st.text_input("Location",
                                           value=getattr(target_device, 'location', ''))

                # Device type (read-only for now)
                device_type = target_device.device_type.value if hasattr(target_device.device_type, 'value') else str(target_device.device_type)
                st.text_input("Device Type", value=device_type, disabled=True)

                # Status
                status_options = ["normal", "maintenance", "anomaly"]
                current_status = target_device.status.value if hasattr(target_device.status, 'value') else str(target_device.status)
                new_status = st.selectbox("Operational Status", status_options,
                                        index=status_options.index(current_status))

            with col2:
                # Technical specifications
                new_manufacturer = st.text_input("Manufacturer",
                                               value=getattr(target_device, 'manufacturer', ''))
                new_model = st.text_input("Model",
                                        value=getattr(target_device, 'model', ''))

                current_power_rating = getattr(target_device, 'power_rating', 0)
                new_power_rating = st.number_input("Power Rating (W)",
                                                 value=current_power_rating, min_value=0)

                # Description
                new_description = st.text_area("Description",
                                             value=getattr(target_device, 'description', ''),
                                             height=100)

            # Save button
            if st.form_submit_button("💾 Save Basic Settings", type="primary"):
                try:
                    from app.models.schemas import DeviceStatus

                    # Update device properties
                    target_device.name = new_name
                    target_device.location = new_location
                    target_device.manufacturer = new_manufacturer
                    target_device.model = new_model
                    target_device.power_rating = new_power_rating
                    target_device.description = new_description
                    target_device.status = DeviceStatus(new_status)

                    # Save to Redis
                    success = redis_service.store_device(target_device)
                    if success:
                        st.success("✅ Basic settings saved successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to save basic settings")
                except Exception as e:
                    st.error(f"❌ Error saving settings: {e}")

    with tab2:
        st.subheader("🔧 Advanced Configuration")

        # Get current metadata
        current_metadata = getattr(target_device, 'metadata', {})

        with st.form(f"advanced_settings_{device_id}"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Operational Parameters**")

                # Energy efficiency rating
                efficiency_options = ["A+++", "A++", "A+", "A", "B", "C", "D", "Unknown"]
                current_efficiency = current_metadata.get('energy_efficiency_rating', 'Unknown')
                new_efficiency = st.selectbox("Energy Efficiency Rating", efficiency_options,
                                            index=efficiency_options.index(current_efficiency) if current_efficiency in efficiency_options else 0)

                # Operating hours
                current_hours = current_metadata.get('operating_hours', 0)
                new_operating_hours = st.number_input("Total Operating Hours",
                                                    value=current_hours, min_value=0)

                # Maintenance schedule
                maintenance_options = ["Weekly", "Monthly", "Quarterly", "Semi-annual", "Annual", "As needed"]
                current_maintenance = current_metadata.get('maintenance_schedule', 'Monthly')
                new_maintenance = st.selectbox("Maintenance Schedule", maintenance_options,
                                             index=maintenance_options.index(current_maintenance) if current_maintenance in maintenance_options else 1)

            with col2:
                st.markdown("**Monitoring Configuration**")

                # Monitoring interval
                current_interval = current_metadata.get('monitoring_interval_seconds', 60)
                new_interval = st.number_input("Monitoring Interval (seconds)",
                                             value=current_interval, min_value=10, max_value=3600)

                # Alert thresholds
                current_threshold = current_metadata.get('power_threshold_kw', 0)
                new_threshold = st.number_input("Power Alert Threshold (kW)",
                                              value=current_threshold, min_value=0.0, step=0.1)

                # Data retention
                retention_options = ["1 day", "1 week", "1 month", "3 months", "1 year"]
                current_retention = current_metadata.get('data_retention', '1 month')
                new_retention = st.selectbox("Data Retention Period", retention_options,
                                           index=retention_options.index(current_retention) if current_retention in retention_options else 2)

            # Advanced features
            st.markdown("**Advanced Features**")
            col3, col4 = st.columns(2)

            with col3:
                enable_auto_shutdown = st.checkbox("Enable Auto-Shutdown",
                                                 value=current_metadata.get('auto_shutdown_enabled', False))
                enable_predictive = st.checkbox("Predictive Maintenance",
                                               value=current_metadata.get('predictive_maintenance', False))

            with col4:
                enable_alerts = st.checkbox("Email Alerts",
                                          value=current_metadata.get('email_alerts', True))
                enable_api = st.checkbox("API Monitoring",
                                       value=current_metadata.get('api_monitoring', True))

            # Save advanced settings
            if st.form_submit_button("🔧 Save Advanced Settings", type="primary"):
                try:
                    # Update metadata
                    new_metadata = {
                        'energy_efficiency_rating': new_efficiency,
                        'operating_hours': new_operating_hours,
                        'maintenance_schedule': new_maintenance,
                        'monitoring_interval_seconds': new_interval,
                        'power_threshold_kw': new_threshold,
                        'data_retention': new_retention,
                        'auto_shutdown_enabled': enable_auto_shutdown,
                        'predictive_maintenance': enable_predictive,
                        'email_alerts': enable_alerts,
                        'api_monitoring': enable_api
                    }

                    target_device.metadata = new_metadata

                    # Save to Redis
                    success = redis_service.store_device(target_device)
                    if success:
                        st.success("✅ Advanced settings saved successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to save advanced settings")
                except Exception as e:
                    st.error(f"❌ Error saving advanced settings: {e}")

    with tab3:
        st.subheader("📊 Monitoring Configuration")

        # Current monitoring status
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Status", target_device.status.value.title())

        with col2:
            metadata = getattr(target_device, 'metadata', {})
            interval = metadata.get('monitoring_interval_seconds', 60)
            st.metric("Monitoring Interval", f"{interval}s")

        with col3:
            threshold = metadata.get('power_threshold_kw', 0)
            st.metric("Alert Threshold", f"{threshold} kW")

        # Monitoring controls
        st.markdown("**Monitoring Actions**")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("▶️ Start Monitoring", key=f"start_monitor_{device_id}"):
                st.success(f"✅ Monitoring started for {device_name}")
                st.info("💡 Monitoring simulated - in production this would trigger real monitoring systems")

        with col2:
            if st.button("⏸️ Pause Monitoring", key=f"pause_monitor_{device_id}"):
                st.warning(f"⏸️ Monitoring paused for {device_name}")

        with col3:
            if st.button("🔄 Reset Counters", key=f"reset_counters_{device_id}"):
                st.info(f"🔄 Counters reset for {device_name}")

        with col4:
            if st.button("📊 Export Data", key=f"export_data_{device_id}"):
                st.info(f"📊 Data export initiated for {device_name}")
                st.info("💡 In production, this would generate and download a CSV/Excel file")

        # Real-time monitoring display
        st.markdown("**Real-time Status**")

        # Simulated real-time data (in production, this would come from actual monitoring)
        import random
        current_time = datetime.now().strftime("%H:%M:%S")
        simulated_power = random.uniform(0.5, 2.0)
        simulated_temp = random.uniform(20, 35)
        simulated_voltage = random.uniform(220, 240)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Power", f"{simulated_power:.2f} kW")

        with col2:
            st.metric("Temperature", f"{simulated_temp:.1f}°C")

        with col3:
            st.metric("Voltage", f"{simulated_voltage:.1f}V")

        with col4:
            st.metric("Last Update", current_time)

    with tab4:
        st.subheader("🚨 Alert Configuration")

        # Current alert settings
        metadata = getattr(target_device, 'metadata', {})

        with st.form(f"alert_settings_{device_id}"):
            st.markdown("**Alert Thresholds**")

            col1, col2 = st.columns(2)

            with col1:
                # Power alerts
                power_threshold = st.number_input("High Power Alert (kW)",
                                                value=metadata.get('power_threshold_kw', 1.0),
                                                min_value=0.0, step=0.1)

                temp_threshold = st.number_input("High Temperature Alert (°C)",
                                               value=metadata.get('temperature_threshold', 50.0),
                                               min_value=0.0, step=1.0)

                voltage_low = st.number_input("Low Voltage Alert (V)",
                                            value=metadata.get('voltage_low_threshold', 200.0),
                                            min_value=0.0, step=1.0)

            with col2:
                # Communication alerts
                offline_threshold = st.number_input("Offline Alert (minutes)",
                                                  value=metadata.get('offline_threshold_minutes', 5),
                                                  min_value=1, max_value=60)

                voltage_high = st.number_input("High Voltage Alert (V)",
                                             value=metadata.get('voltage_high_threshold', 250.0),
                                             min_value=0.0, step=1.0)

                efficiency_threshold = st.number_input("Low Efficiency Alert (%)",
                                                     value=metadata.get('efficiency_threshold', 70.0),
                                                     min_value=0.0, max_value=100.0, step=1.0)

            st.markdown("**Notification Settings**")

            col3, col4 = st.columns(2)

            with col3:
                email_alerts = st.checkbox("Email Notifications",
                                         value=metadata.get('email_alerts', True))
                sms_alerts = st.checkbox("SMS Notifications",
                                       value=metadata.get('sms_alerts', False))

            with col4:
                dashboard_alerts = st.checkbox("Dashboard Alerts",
                                             value=metadata.get('dashboard_alerts', True))
                api_webhooks = st.checkbox("API Webhooks",
                                         value=metadata.get('api_webhooks', False))

            # Alert recipients
            st.markdown("**Alert Recipients**")
            current_emails = metadata.get('alert_emails', 'admin@example.com')
            alert_emails = st.text_area("Email Recipients (comma-separated)",
                                       value=current_emails, height=60)

            # Save alert settings
            if st.form_submit_button("🚨 Save Alert Settings", type="primary"):
                try:
                    # Update alert metadata
                    alert_metadata = {
                        **metadata,  # Keep existing metadata
                        'power_threshold_kw': power_threshold,
                        'temperature_threshold': temp_threshold,
                        'voltage_low_threshold': voltage_low,
                        'voltage_high_threshold': voltage_high,
                        'efficiency_threshold': efficiency_threshold,
                        'offline_threshold_minutes': offline_threshold,
                        'email_alerts': email_alerts,
                        'sms_alerts': sms_alerts,
                        'dashboard_alerts': dashboard_alerts,
                        'api_webhooks': api_webhooks,
                        'alert_emails': alert_emails
                    }

                    target_device.metadata = alert_metadata

                    # Save to Redis
                    success = redis_service.store_device(target_device)
                    if success:
                        st.success("✅ Alert settings saved successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to save alert settings")
                except Exception as e:
                    st.error(f"❌ Error saving alert settings: {e}")

        # Test alerts
        st.markdown("**Test Alerts**")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📧 Test Email Alert", key=f"test_email_{device_id}"):
                st.success("✅ Test email sent!")
                st.info("💡 In production, this would send an actual test email")

        with col2:
            if st.button("📱 Test SMS Alert", key=f"test_sms_{device_id}"):
                st.success("✅ Test SMS sent!")
                st.info("💡 In production, this would send an actual test SMS")

        with col3:
            if st.button("🔔 Test Dashboard Alert", key=f"test_dashboard_{device_id}"):
                st.success("✅ Dashboard alert triggered!")
                st.info("💡 This would appear in the main dashboard alerts section")

    # Close button
    if st.button("❌ Close Settings", key="close_settings"):
        del st.session_state.device_to_configure
        st.rerun()

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">⚡ Redisense Energy Monitoring Dashboard</h1>', unsafe_allow_html=True)

    # Navigation Menu in Sidebar
    st.sidebar.title("📋 Navigation")

    # Navigation options
    nav_options = {
        "🏠 Main Dashboard": "main_dashboard",
        "📈 Energy Consumption Trends": "energy_trends",
        "🔍 Semantic Device Search": "semantic_search",
        "🏠 Device Status": "device_status",
        "📋 Show Raw Data": "raw_data"
    }

    # Create navigation menu
    selected_section = st.sidebar.radio(
        "Navigate to:",
        options=list(nav_options.keys()),
        index=0  # Default to Main Dashboard
    )

    # Get the section key
    section_key = nav_options[selected_section]

    # Clear any search-related query parameters if not in semantic search section
    if section_key != "semantic_search":
        try:
            if 'search' in st.query_params:
                del st.query_params['search']
        except:
            pass  # Ignore any query param errors

    st.sidebar.markdown("---")

    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (3min)", value=True)

    if auto_refresh:
        # Auto-refresh every 3 minutes
        time.sleep(0.1)  # Small delay to prevent too frequent updates

    # Refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()

    # Load data
    devices, readings, success, error = load_data()

    if not success:
        st.error(f"❌ Error loading data: {error}")
        st.info("Please check your Redis connection and ensure data has been seeded.")
        st.code("uv run python cli.py test-connection")
        st.code("uv run python cli.py seed-data --device-count 5 --days 2")
        return

    if not devices:
        st.warning("⚠️ No devices found in the system.")
        st.info("Seed some data to get started:")
        st.code("uv run python cli.py seed-data --device-count 5 --days 2")
        return

    # Calculate metrics
    metrics = create_real_time_metrics(readings)

    # Display content based on selected section
    if section_key == "main_dashboard":
        display_main_dashboard(devices, readings, metrics)
    elif section_key == "energy_trends":
        display_energy_trends(devices, readings, metrics)
    elif section_key == "semantic_search":
        # Only display semantic search when explicitly selected
        display_semantic_search()
    elif section_key == "device_status":
        display_device_status(devices, metrics)
    elif section_key == "raw_data":
        display_raw_data(readings)
    else:
        # Default to main dashboard if no valid section
        display_main_dashboard(devices, readings, metrics)

    # Device analysis section (if a device is selected for analysis)
    try:
        display_device_analysis(None, readings, devices, metrics)
    except Exception as e:
        if 'device_to_analyze' in st.session_state:
            st.error(f"Error displaying device analysis: {e}")
            del st.session_state.device_to_analyze

    # Device settings section (if a device is selected for configuration)
    try:
        settings = Settings()
        redis_service = RedisService(settings)
        display_device_settings(None, devices, redis_service)
    except Exception as e:
        if 'device_to_configure' in st.session_state:
            st.error(f"Error displaying device settings: {e}")
            del st.session_state.device_to_configure

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Redisense Energy Monitoring System | Powered by Redis + AI"
        "</div>",
        unsafe_allow_html=True
    )

    # Auto-refresh
    if auto_refresh:
        time.sleep(180)  # 3 minutes
        st.rerun()

def display_main_dashboard(devices, readings, metrics):
    """Display the main dashboard with overview metrics and charts"""
    st.header("🏠 Main Dashboard")

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🏠 Total Devices",
            value=metrics['total_devices'],
            delta=None
        )

    with col2:
        st.metric(
            label="⚡ Current Power",
            value=f"{metrics['current_power']:.2f} kW",
            delta=None
        )

    with col3:
        st.metric(
            label="📊 Avg Consumption",
            value=f"{metrics['avg_consumption']:.2f} kW",
            delta=None
        )

    with col4:
        st.metric(
            label="📈 24h Total Energy",
            value=f"{metrics['total_energy_24h']:.2f} kWh",
            delta=None
        )

    # Charts section
    st.markdown("---")

    # Device overview
    col1, col2 = st.columns([1, 1])

    with col1:
        device_overview_fig = create_device_overview_chart(devices)
        if device_overview_fig:
            st.plotly_chart(device_overview_fig, use_container_width=True)

    with col2:
        # System status
        st.subheader("🔧 System Status")

        if readings:
            latest_reading_time = max(r.timestamp for r in readings)
            time_diff = datetime.utcnow() - latest_reading_time

            if time_diff.total_seconds() < 300:  # Less than 5 minutes
                st.success("✅ System is actively receiving data")
            elif time_diff.total_seconds() < 3600:  # Less than 1 hour
                st.warning("⚠️ Data is slightly outdated")
            else:
                st.error("❌ Data is significantly outdated")

            st.info(f"Latest reading: {latest_reading_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.error("❌ No recent readings found")

        # Quick actions
        st.subheader("🚀 Quick Actions")
        if st.button("Start Data Streaming"):
            st.info("Run in terminal: `uv run python cli.py stream-data --device-count 5 --interval 10`")

        if st.button("Clear All Data"):
            st.warning("Run in terminal: `uv run python cli.py clear-data`")

def display_energy_trends(devices, readings, metrics):
    """Display energy consumption trends and patterns"""
    st.header("📈 Energy Consumption Trends")

    # Time range filter - positioned above the chart for full width
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        time_range_options = {
            "24 Hours": 24,
            "3 Hours": 3,
            "1 Hour": 1,
            "30 Minutes": 0.5,
            "5 Minutes": 5/60  # 5 minutes as fraction of hour
        }
        selected_range = st.selectbox(
            "Time Range",
            options=list(time_range_options.keys()),
            index=0  # Default to 24 Hours
        )
        time_range_hours = time_range_options[selected_range]

    # Chart takes full width
    energy_trends_fig = create_energy_trends_chart(readings, devices, time_range_hours)
    if energy_trends_fig:
        st.plotly_chart(energy_trends_fig, use_container_width=True)
    else:
        st.info(f"No energy trend data available for the last {selected_range.lower()}. Start streaming data to see real-time trends.")

    # Additional analytics
    st.markdown("---")
    st.subheader("📊 Energy Analytics")

    if readings:
        # Calculate some analytics
        total_energy = sum(r.energy_kwh for r in readings)
        avg_power = sum(r.power_kw or r.energy_kwh for r in readings) / len(readings)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Energy Consumed", f"{total_energy:.2f} kWh")

        with col2:
            st.metric("Average Power", f"{avg_power:.2f} kW")

        with col3:
            # Calculate peak usage time
            if len(readings) > 0:
                hourly_usage = {}
                for reading in readings:
                    hour = reading.timestamp.hour
                    if hour not in hourly_usage:
                        hourly_usage[hour] = []
                    hourly_usage[hour].append(reading.power_kw or reading.energy_kwh)

                if hourly_usage:
                    peak_hour = max(hourly_usage.keys(), key=lambda h: sum(hourly_usage[h]) / len(hourly_usage[h]))
                    st.metric("Peak Usage Hour", f"{peak_hour}:00")

def display_semantic_search():
    """Display the semantic search interface"""
    st.header("🔍 Semantic Device Search")
    st.markdown("*Powered by Redis 8 Vector Search with AI embeddings*")

    # Create search interface
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input(
            "Search devices using natural language:",
            placeholder="Try: 'HVAC system', 'air conditioning', 'energy efficient lighting', 'server cooling'...",
            help="Use natural language to find devices semantically. The AI understands context and finds related equipment."
        )

    with col2:
        search_limit = st.selectbox("Results:", [3, 5, 10], index=0)
        search_button = st.button("🔍 Search", type="primary")

    # Perform semantic search
    if search_query and (search_button or search_query):
        try:
            with st.spinner("🧠 Performing semantic search..."):
                settings = Settings()
                ai_service = AIService()

                # Perform semantic search
                search_results = ai_service.semantic_search(search_query, search_limit)

                if search_results:
                    st.success(f"✅ Found {len(search_results)} semantic matches!")

                    # Display search results in an attractive format
                    for i, result in enumerate(search_results, 1):
                        with st.expander(f"🎯 Match #{i} - Device: {result['device_id']} (Score: {result['score']:.3f})", expanded=True):

                            # Create columns for better layout
                            result_col1, result_col2 = st.columns([2, 1])

                            with result_col1:
                                st.markdown(f"**📄 Content Preview:**")
                                st.write(result['content'][:200] + "..." if len(result['content']) > 200 else result['content'])

                                # Similarity score visualization
                                score_percentage = result['score'] * 100
                                st.progress(result['score'], f"Similarity: {score_percentage:.1f}%")

                            with result_col2:
                                metadata = result.get('metadata', {})
                                if metadata:
                                    st.markdown("**🏷️ Device Info:**")
                                    if metadata.get('device_type'):
                                        st.write(f"**Type:** {metadata['device_type']}")
                                    if metadata.get('manufacturer'):
                                        st.write(f"**Manufacturer:** {metadata['manufacturer']}")
                                    if metadata.get('location'):
                                        st.write(f"**Location:** {metadata['location']}")
                else:
                    st.warning("🔍 No semantic matches found. Try different keywords like 'HVAC', 'cooling', 'lighting', etc.")

        except Exception as e:
            st.error(f"❌ Search error: {str(e)}")
            st.info("💡 Make sure vector embeddings are generated: `uv run python cli.py generate-embeddings`")

    # Show example searches
    if not search_query:
        st.markdown("### 💡 Try these example searches:")
        example_col1, example_col2, example_col3 = st.columns(3)

        with example_col1:
            if st.button("🌡️ HVAC System"):
                st.session_state['example_search'] = "HVAC system"
                st.rerun()

        with example_col2:
            if st.button("💡 Energy Efficient"):
                st.session_state['example_search'] = "energy efficient"
                st.rerun()

        with example_col3:
            if st.button("🖥️ Server Cooling"):
                st.session_state['example_search'] = "server cooling"
                st.rerun()

    # Handle example search clicks
    if 'example_search' in st.session_state:
        search_query = st.session_state['example_search']
        del st.session_state['example_search']
        # Re-render with the example search
        st.rerun()

def display_device_status(devices, metrics):
    """Display device status cards and similar device functionality"""
    st.header("🏠 Device Status")

    # Ensure no search UI elements are accidentally rendered
    # Clear any potential search-related session state
    if 'search_query' in st.session_state:
        del st.session_state['search_query']

    # Device cards
    create_device_cards(devices, metrics)

    # Similar devices section
    display_similar_devices_section(devices)

def display_raw_data(readings):
    """Display raw data table"""
    st.header("📋 Raw Energy Readings Data")

    if readings:
        # Convert to DataFrame for display
        df_display = []
        for reading in readings[-50:]:  # Last 50 readings
            df_display.append({
                'Device ID': reading.device_id,
                'Timestamp': reading.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Energy (kWh)': f"{reading.energy_kwh:.3f}",
                'Power (kW)': f"{reading.power_kw:.3f}" if reading.power_kw else "N/A",
                'Voltage (V)': f"{reading.voltage:.1f}" if reading.voltage else "N/A",
                'Current (A)': f"{reading.current:.2f}" if reading.current else "N/A"
            })

        df_display = pd.DataFrame(df_display)

        # Add some statistics
        st.subheader("📊 Data Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Readings", len(readings))

        with col2:
            st.metric("Displayed Readings", len(df_display))

        with col3:
            if readings:
                time_span = max(r.timestamp for r in readings) - min(r.timestamp for r in readings)
                st.metric("Time Span", f"{time_span.total_seconds() / 3600:.1f} hours")

        st.subheader("📋 Recent Readings")
        st.dataframe(df_display, use_container_width=True)

        # Export functionality
        st.subheader("📤 Export Data")
        if st.button("📊 Export to CSV"):
            st.info("💡 In production, this would generate and download a CSV file with all readings")

    else:
        st.info("No readings to display. Start streaming data to see real-time readings.")
        st.code("uv run python cli.py stream-data --device-count 5 --interval 10")

if __name__ == "__main__":
    main()
