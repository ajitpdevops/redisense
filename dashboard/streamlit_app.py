import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings
from app.services.redis_service import RedisService
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

    # Add device profile selection in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏠 Device Profiles")

    device_options = ["Select a device..."] + [f"{getattr(device, 'name', device.device_id)} ({device.device_id})" for device in devices]
    selected_device = st.sidebar.selectbox("View Device Profile", device_options)

    if selected_device != "Select a device...":
        device_id = selected_device.split('(')[1].split(')')[0]
        selected_device_obj = next((d for d in devices if d.device_id == device_id), None)

        if selected_device_obj:
            st.sidebar.markdown("### 📋 Device Details")
            st.sidebar.markdown(f"**Name:** {getattr(selected_device_obj, 'name', selected_device_obj.device_id)}")
            st.sidebar.markdown(f"**Type:** {selected_device_obj.device_type.value}")
            st.sidebar.markdown(f"**Manufacturer:** {getattr(selected_device_obj, 'manufacturer', 'Unknown')}")
            st.sidebar.markdown(f"**Model:** {getattr(selected_device_obj, 'model', 'Unknown')}")
            st.sidebar.markdown(f"**Location:** {getattr(selected_device_obj, 'location', 'Unknown')}")
            st.sidebar.markdown(f"**Power Rating:** {getattr(selected_device_obj, 'power_rating', 'Unknown')}W")
            st.sidebar.markdown(f"**Status:** {selected_device_obj.status.value.title()}")

            description = getattr(selected_device_obj, 'description', None)
            if description:
                st.sidebar.markdown(f"**Description:** {description}")

            # Show metadata if available
            metadata = getattr(selected_device_obj, 'metadata', {})
            if metadata:
                st.sidebar.markdown("### 🔧 Technical Info")
                if 'energy_efficiency_rating' in metadata:
                    st.sidebar.markdown(f"**Efficiency:** {metadata['energy_efficiency_rating']}")
                if 'maintenance_schedule' in metadata:
                    st.sidebar.markdown(f"**Maintenance:** {metadata['maintenance_schedule']}")
                if 'operating_hours' in metadata:
                    st.sidebar.markdown(f"**Operating Hours:** {metadata['operating_hours']:,}")
                if 'last_maintenance' in metadata:
                    st.sidebar.markdown(f"**Last Service:** {metadata['last_maintenance']}")

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
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📊 Analyze", key=f"analyze_{device.device_id}"):
                            st.info(f"Analysis for {device_name} would be displayed here")
                    with col2:
                        if st.button(f"⚙️ Settings", key=f"settings_{device.device_id}"):
                            st.info(f"Settings for {device_name} would be displayed here")

def create_device_editor_sidebar(devices, redis_service):
    """Create device profile editor in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("✏️ Edit Device Profile")

    if not devices:
        st.sidebar.info("No devices to edit")
        return

    # Device selection for editing
    device_options = ["Select device to edit..."] + [f"{getattr(device, 'name', device.device_id)} ({device.device_id})" for device in devices]
    selected_edit_device = st.sidebar.selectbox("Select Device to Edit", device_options, key="edit_device")

    if selected_edit_device != "Select device to edit...":
        device_id = selected_edit_device.split('(')[1].split(')')[0]
        device = next((d for d in devices if d.device_id == device_id), None)

        if device:
            st.sidebar.markdown("### 📝 Edit Properties")

            # Create edit form
            with st.sidebar.form(f"edit_form_{device_id}"):
                current_name = getattr(device, 'name', device.device_id)
                current_description = getattr(device, 'description', '')
                current_location = getattr(device, 'location', '')

                new_name = st.text_input("Device Name", value=current_name)
                new_description = st.text_area("Description", value=current_description, height=100)
                new_location = st.text_input("Location", value=current_location)

                status_options = ["normal", "maintenance", "anomaly"]
                current_status = device.status.value if hasattr(device.status, 'value') else str(device.status)
                new_status = st.selectbox("Status", status_options, index=status_options.index(current_status))

                submitted = st.form_submit_button("💾 Save Changes")

                if submitted:
                    try:
                        # Update device
                        from app.models.schemas import DeviceStatus
                        device.name = new_name
                        device.description = new_description
                        device.location = new_location
                        device.status = DeviceStatus(new_status)

                        # Save to Redis
                        success = redis_service.store_device(device)
                        if success:
                            st.sidebar.success("✅ Device updated successfully!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.sidebar.error("❌ Failed to update device")
                    except Exception as e:
                        st.sidebar.error(f"❌ Error: {e}")

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">⚡ Redisense Energy Monitoring Dashboard</h1>', unsafe_allow_html=True)

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

    # Add device profile editor to sidebar
    try:
        settings = Settings()
        redis_service = RedisService(settings)
        create_device_editor_sidebar(devices, redis_service)
    except:
        pass  # Don't let editor errors break the main dashboard

    # Calculate metrics
    metrics = create_real_time_metrics(readings)

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

    # Energy trends
    st.markdown("---")
    st.subheader("📈 Energy Consumption Trends")

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

    # Device cards
    st.markdown("---")
    st.subheader("🏠 Device Status")
    create_device_cards(devices, metrics)

    # Data table
    if st.sidebar.checkbox("Show Raw Data"):
        st.markdown("---")
        st.subheader("📋 Recent Energy Readings")

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
            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("No readings to display")

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

if __name__ == "__main__":
    main()
