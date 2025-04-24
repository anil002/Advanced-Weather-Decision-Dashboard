import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import io
from uuid import uuid4
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# Streamlit page configuration
st.set_page_config(page_title="Advanced Weather Decision Dashboard", layout="wide")

# Title
st.title("Advanced Weather Decision Dashboard")
st.markdown("Developed by: **Dr. Anil Kumar Singh**")
st.markdown("**Email:singhanil854@gmail.com**")

# Available variables
VARIABLES = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'precipitation', 'cloud_cover', 'pressure', 'surface_temperature', 'dew_point']
GRAPH_TYPES = ['Line', 'Histogram', 'Scatter', 'Bar', 'Box', 'Violin']

# Initialize session state for latitude and longitude
if 'latitude' not in st.session_state:
    st.session_state['latitude'] = 28.70
if 'longitude' not in st.session_state:
    st.session_state['longitude'] = 77.10

# Sidebar for inputs
st.sidebar.header("Data Selection")
data_type = st.sidebar.selectbox("Data Type", ["Hourly", "Daily", "Historical"])

# Numeric inputs for latitude and longitude
st.sidebar.header("Location Input")
latitude_input = st.sidebar.number_input("Latitude", value=st.session_state['latitude'], min_value=-90.0, max_value=90.0, step=0.01, key="lat_input")
longitude_input = st.sidebar.number_input("Longitude", value=st.session_state['longitude'], min_value=-180.0, max_value=180.0, step=0.01, key="lon_input")

# Update session state if numeric inputs change
if latitude_input != st.session_state['latitude']:
    st.session_state['latitude'] = latitude_input
if longitude_input != st.session_state['longitude']:
    st.session_state['longitude'] = longitude_input

# Map-based location selector
st.sidebar.header("Map Location Selector")
st.sidebar.markdown("""
**Purpose**: Select a location by clicking on the map or entering coordinates manually.
**How to Use**: Click a point on the map to set latitude and longitude, or use the numeric inputs above. The selected coordinates are displayed below the map.
""")
m = folium.Map(location=[st.session_state['latitude'], st.session_state['longitude']], zoom_start=10)
folium.Marker([st.session_state['latitude'], st.session_state['longitude']], popup="Selected Location").add_to(m)

# Add click event handler
m.add_child(folium.LatLngPopup())
map_data = st_folium(m, width=300, height=300)

# Update session state if map is clicked
if map_data.get("last_clicked"):
    st.session_state['latitude'] = map_data["last_clicked"]["lat"]
    st.session_state['longitude'] = map_data["last_clicked"]["lng"]

# Display selected coordinates
st.sidebar.write(f"Selected: Lat {st.session_state['latitude']:.2f}, Lon {st.session_state['longitude']:.2f}")

# Date range for historical data
if data_type == "Historical":
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    date_range = st.sidebar.date_input("Select Date Range", [start_date, end_date], min_value=datetime(2010, 1, 1).date(), max_value=end_date)
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = start_date, end_date

# Fetch data button
if st.sidebar.button("Fetch Weather Data"):
    with st.spinner("Fetching data..."):
        try:
            # Construct API URL
            variables = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover,pressure_msl,surface_temperature,dew_point_2m"
            if data_type == "Historical":
                url = f"https://archive-api.open-meteo.com/v1/archive?latitude={st.session_state['latitude']}&longitude={st.session_state['longitude']}&start_date={start_date}&end_date={end_date}&hourly={variables}"
            else:
                url = f"https://api.open-meteo.com/v1/forecast?latitude={st.session_state['latitude']}&longitude={st.session_state['longitude']}&{'hourly' if data_type == 'Hourly' else 'daily'}={variables}&time_mode=time_interval&start_date=2025-04-16&end_date=2025-04-30&models=best_match"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # Process data into DataFrame with type enforcement
            time_key = 'hourly' if data_type in ['Hourly', 'Historical'] else 'daily'
            df = pd.DataFrame({
                'time': pd.to_datetime(data[time_key]['time']),
                'temperature': pd.to_numeric(data[time_key]['temperature_2m'], errors='coerce').astype('float64'),
                'humidity': pd.to_numeric(data[time_key]['relative_humidity_2m'], errors='coerce').astype('float64'),
                'wind_speed': pd.to_numeric(data[time_key]['wind_speed_10m'], errors='coerce').astype('float64'),
                'wind_direction': pd.to_numeric(data[time_key]['wind_direction_10m'], errors='coerce').astype('float64'),
                'precipitation': pd.to_numeric(data[time_key]['precipitation'], errors='coerce').astype('float64'),
                'cloud_cover': pd.to_numeric(data[time_key]['cloud_cover'], errors='coerce').astype('float64'),
                'pressure': pd.to_numeric(data[time_key]['pressure_msl'], errors='coerce').astype('float64'),
                'surface_temperature': pd.to_numeric(data[time_key]['surface_temperature'], errors='coerce').astype('float64'),
                'dew_point': pd.to_numeric(data[time_key]['dew_point_2m'], errors='coerce').astype('float64')
            })

            # Handle missing values
            df.fillna(method='ffill', inplace=True)
            if df.isna().any().any():
                df.fillna(0, inplace=True)

            # Cache data
            st.session_state['df'] = df
            st.session_state['data_type'] = data_type
            st.success("Data fetched successfully!")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

# Function to create wind rose data
def create_wind_rose_data(df, direction_col='wind_direction', speed_col='wind_speed', bins=16):
    bin_size = 360 / bins
    df['direction_bin'] = np.round(df[direction_col] / bin_size) * bin_size
    df['direction_bin'] = df['direction_bin'] % 360
    speed_bins = np.histogram(df[speed_col], bins=5)[1]
    df['speed_bin'] = pd.cut(df[speed_col], bins=speed_bins, labels=[f"{speed_bins[i]:.1f}-{speed_bins[i+1]:.1f}" for i in range(len(speed_bins)-1)], include_lowest=True)
    wind_rose_data = df.groupby(['direction_bin', 'speed_bin']).size().reset_index(name='count')
    return wind_rose_data, speed_bins

# Function to generate PDF report
def generate_pdf_report(df, alerts, decision_matrix):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Weather Decision Report")
    y = 700
    c.drawString(100, y, f"Location: Lat {st.session_state['latitude']:.2f}, Lon {st.session_state['longitude']:.2f}")
    y -= 20
    c.drawString(100, y, f"Data Type: {st.session_state['data_type']}")
    y -= 30
    c.drawString(100, y, "Key Metrics:")
    y -= 20
    for var in VARIABLES:
        mean = df[var].mean()
        c.drawString(100, y, f"{var.capitalize()}: Mean = {mean:.2f}")
        y -= 20
    y -= 20
    c.drawString(100, y, "Alerts:")
    y -= 20
    for alert in alerts:
        c.drawString(100, y, alert)
        y -= 20
    y -= 20
    c.drawString(100, y, "Decision Matrix:")
    y -= 20
    for decision in decision_matrix:
        c.drawString(100, y, decision)
        y -= 20
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Check if data is available
if 'df' in st.session_state:
    df = st.session_state['df']
    data_type = st.session_state['data_type']
    
    # Summary Dashboard
    st.subheader("Summary Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Temperature", f"{df['temperature'].mean():.1f} °C")
    with col2:
        st.metric("Total Precipitation", f"{df['precipitation'].sum():.1f} mm")
    with col3:
        st.metric("Avg Wind Speed", f"{df['wind_speed'].mean():.1f} km/h")

    # Weather Alerts
    st.subheader("Weather Alerts")
    with st.expander("Configure Alerts"):
        st.markdown("""
        **Purpose**: Generates alerts based on user-defined thresholds for critical weather variables.
        **How to Use**: Set thresholds for temperature, precipitation, or wind speed. Alerts are triggered when values exceed these limits.
        """)
        temp_threshold = st.number_input("Temperature Threshold (°C)", value=30.0)
        precip_threshold = st.number_input("Precipitation Threshold (mm)", value=10.0)
        wind_threshold = st.number_input("Wind Speed Threshold (km/h)", value=50.0)
    
    alerts = []
    if df['temperature'].max() > temp_threshold:
        alerts.append(f"High Temperature Alert: Max {df['temperature'].max():.1f} °C exceeds {temp_threshold} °C")
    if df['precipitation'].max() > precip_threshold:
        alerts.append(f"Heavy Precipitation Alert: Max {df['precipitation'].max():.1f} mm exceeds {precip_threshold} mm")
    if df['wind_speed'].max() > wind_threshold:
        alerts.append(f"High Wind Speed Alert: Max {df['wind_speed'].max():.1f} km/h exceeds {wind_threshold} km/h")
    
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.info("No weather alerts triggered.")

    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Statistical Summary
    with st.expander("Statistical Summary"):
        st.markdown("""
        **Purpose**: Provides descriptive statistics for selected weather variables.
        **How to Use**: Select variables to view mean, std, min, max, and quartiles for planning and analysis.
        """)
        selected_vars_stats = st.multiselect("Select Variables for Summary", VARIABLES, default=VARIABLES)
        if selected_vars_stats:
            stats = df[selected_vars_stats].describe().transpose()
            stats['std'] = df[selected_vars_stats].std()
            st.dataframe(stats)
        else:
            st.warning("Please select at least one variable.")

    # Visualizations
    st.subheader("Visualizations")

    # Time Series Plot
    with st.expander("Time Series Plot"):
        st.markdown("""
        **Purpose**: Tracks trends over time for decision-making in scheduling or resource allocation.
        **How to Use**: Select variables and graph type. Line for trends, Histogram for distributions, Box/Violin for spread.
        """)
        selected_vars_ts = st.multiselect("Select Variables for Time Series", VARIABLES, default=['temperature'], key="ts_vars")
        graph_type_ts = st.selectbox("Select Graph Type", GRAPH_TYPES, key="ts_graph")
        if selected_vars_ts:
            try:
                if graph_type_ts == "Line":
                    fig = px.line(df, x='time', y=selected_vars_ts, title=f"{data_type} Time Series")
                elif graph_type_ts == "Histogram":
                    fig = px.histogram(df, x=selected_vars_ts, title="Histogram")
                elif graph_type_ts == "Scatter":
                    fig = px.scatter(df, x='time', y=selected_vars_ts, title="Scatter Plot")
                elif graph_type_ts == "Bar":
                    fig = go.Figure(data=[go.Bar(x=df['time'], y=df[var], name=var) for var in selected_vars_ts])
                    fig.update_layout(title="Bar Plot", barmode='group')
                elif graph_type_ts == "Box":
                    fig = px.box(df, y=selected_vars_ts, title="Box Plot")
                elif graph_type_ts == "Violin":
                    fig = px.violin(df, y=selected_vars_ts, box=True, points="all", title="Violin Plot")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in plot: {str(e)}")
        else:
            st.warning("Please select at least one variable.")

    # Aggregated Plot
    with st.expander("Aggregated Plot"):
        st.markdown("""
        **Purpose**: Summarizes daily or time-based averages for strategic planning.
        **How to Use**: Select variables and graph type. Bar for comparisons, Box/Violin for distributions.
        """)
        selected_vars_agg = st.multiselect("Select Variables for Aggregated Plot", VARIABLES, default=['temperature'], key="agg_vars")
        graph_type_agg = st.selectbox("Select Graph Type", GRAPH_TYPES, key="agg_graph")
        if selected_vars_agg:
            agg_col = 'time' if data_type == 'Daily' else df['time'].dt.date
            df_agg = df.groupby(agg_col)[selected_vars_agg].mean().reset_index()
            try:
                if graph_type_agg == "Line":
                    fig = px.line(df_agg, x='time', y=selected_vars_agg, title=f"{data_type} Aggregated")
                elif graph_type_agg == "Histogram":
                    fig = px.histogram(df_agg, x=selected_vars_agg, title="Histogram")
                elif graph_type_agg == "Scatter":
                    fig = px.scatter(df_agg, x='time', y=selected_vars_agg, title="Scatter Plot")
                elif graph_type_agg == "Bar":
                    fig = go.Figure(data=[go.Bar(x=df_agg['time'], y=df_agg[var], name=var) for var in selected_vars_agg])
                    fig.update_layout(title="Bar Plot", barmode='group')
                elif graph_type_agg == "Box":
                    fig = px.box(df_agg, y=selected_vars_agg, title="Box Plot")
                elif graph_type_agg == "Violin":
                    fig = px.violin(df_agg, y=selected_vars_agg, box=True, points="all", title="Violin Plot")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in plot: {str(e)}")
        else:
            st.warning("Please select at least one variable.")

    # Multi-Variable Comparison
    with st.expander("Multi-Variable Comparison"):
        st.markdown("""
        **Purpose**: Compares multiple variables in a single plot for integrated decision-making (e.g., temperature vs. precipitation).
        **How to Use**: Select two variables for X and Y axes and a graph type. Scatter for correlations, Line for trends.
        """)
        x_var = st.selectbox("Select X Variable", VARIABLES, key="comp_x")
        y_var = st.selectbox("Select Y Variable", VARIABLES, key="comp_y")
        graph_type_comp = st.selectbox("Select Graph Type", ['Scatter', 'Line'], key="comp_graph")
        if x_var and y_var:
            try:
                if graph_type_comp == "Scatter":
                    fig = px.scatter(df, x=x_var, y=y_var, color='time', title=f"{x_var.capitalize()} vs {y_var.capitalize()}")
                elif graph_type_comp == "Line":
                    fig = px.line(df, x=x_var, y=y_var, title=f"{x_var.capitalize()} vs {y_var.capitalize()}")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error in plot: {str(e)}")

    # Wind Rose Plot
    with st.expander("Wind Rose Plot"):
        st.markdown("""
        **Purpose**: Analyzes wind direction and speed distribution for aviation, construction, or renewable energy planning.
        **How to Use**: Available for Hourly/Historical data. Adjust direction bins for granularity. Colors indicate speed ranges.
        """)
        if data_type in ['Hourly', 'Historical']:
            bins = st.slider("Number of Direction Bins", 8, 36, 16, step=4, key="wind_rose_bins")
            wind_rose_data, speed_bins = create_wind_rose_data(df, bins=bins)
            fig_wind_rose = go.Figure()
            for speed_bin in wind_rose_data['speed_bin'].unique():
                df_bin = wind_rose_data[wind_rose_data['speed_bin'] == speed_bin]
                fig_wind_rose.add_trace(go.Barpolar(
                    r=df_bin['count'],
                    theta=df_bin['direction_bin'],
                    name=speed_bin,
                    opacity=0.8
                ))
            fig_wind_rose.update_layout(
                title="Wind Rose",
                polar=dict(
                    radialaxis=dict(showticklabels=True),
                    angularaxis=dict(direction="clockwise", period=360)
                ),
                showlegend=True
            )
            st.plotly_chart(fig_wind_rose, use_container_width=True)
        else:
            st.info("Wind Rose is available for Hourly or Historical data only.")

    # Heatmap
    with st.expander("Hourly Trends Heatmap"):
        st.markdown("""
        **Purpose**: Identifies daily cycles in a variable for scheduling or operational planning.
        **How to Use**: Choose one variable to visualize hourly patterns across days (Hourly/Historical data).
        """)
        selected_var_heat = st.selectbox("Select Variable for Heatmap", VARIABLES, key="heat_var")
        if data_type in ['Hourly', 'Historical']:
            df_heatmap = df.pivot_table(index=df['time'].dt.hour, columns=df['time'].dt.date,
                                        values=selected_var_heat, aggfunc='mean')
            fig_heat = px.imshow(df_heatmap, title=f"Hourly {selected_var_heat.capitalize()} Heatmap",
                                 labels=dict(x="Date", y="Hour", color=f"{selected_var_heat.capitalize()}"))
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Heatmap is available for Hourly or Historical data only.")

    # Correlation Matrix
    with st.expander("Correlation Matrix"):
        st.markdown("""
        **Purpose**: Identifies relationships between variables for risk assessment or forecasting.
        **How to Use**: Select multiple variables to view correlation coefficients. Values near 1 or -1 indicate strong relationships.
        """)
        selected_vars_corr = st.multiselect("Select Variables for Correlation", VARIABLES, default=VARIABLES, key="corr_vars")
        if selected_vars_corr and len(selected_vars_corr) > 1:
            corr_matrix = df[selected_vars_corr].corr()
            fig_corr, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            plt.title("Correlation Matrix")
            st.pyplot(fig_corr)
        else:
            st.warning("Please select at least two variables.")

    # Advanced Analytics
    st.subheader("Advanced Analytics")

    # Rolling Mean
    with st.expander("Trend Analysis (Rolling Mean)"):
        st.markdown("""
        **Purpose**: Smooths data to reveal long-term trends for strategic planning.
        **How to Use**: Select variables and adjust window size. Larger windows reduce noise.
        """)
        selected_vars_roll = st.multiselect("Select Variables for Rolling Mean", VARIABLES, default=['temperature'], key="roll_vars")
        window = st.slider("Select rolling window", 1, 24 if data_type in ['Hourly', 'Historical'] else 7, 12, key="roll_window")
        if selected_vars_roll:
            df_rolling = df.rolling(window=window, on='time')[selected_vars_roll].mean()
            fig_trend = px.line(df_rolling, x=df['time'], y=selected_vars_roll,
                                title=f"Rolling Mean (Window: {window} {'hours' if data_type in ['Hourly', 'Historical'] else 'days'})")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("Please select at least one variable.")

    # Time Series Decomposition
    with st.expander("Time Series Decomposition"):
        st.markdown("""
        **Purpose**: Breaks down a variable into trend, seasonal, and residual components for pattern analysis.
        **How to Use**: Select one variable. Requires 48+ points (Hourly/Historical data).
        """)
        selected_var_decomp = st.selectbox("Select Variable for Decomposition", VARIABLES, key="decomp_var")
        if data_type in ['Hourly', 'Historical'] and len(df) >= 48:
            period = 24 if data_type == 'Hourly' else 7
            decomposition = seasonal_decompose(df[selected_var_decomp], model='additive', period=period)
            fig_decomp = go.Figure()
            fig_decomp.add_trace(go.Scatter(x=df['time'], y=decomposition.observed, name='Observed'))
            fig_decomp.add_trace(go.Scatter(x=df['time'], y=decomposition.trend, name='Trend'))
            fig_decomp.add_trace(go.Scatter(x=df['time'], y=decomposition.seasonal, name='Seasonal'))
            fig_decomp.add_trace(go.Scatter(x=df['time'], y=decomposition.resid, name='Residual'))
            fig_decomp.update_layout(title=f"{selected_var_decomp.capitalize()} Decomposition")
            st.plotly_chart(fig_decomp, use_container_width=True)
        else:
            st.info("Decomposition requires Hourly or Historical data with 48+ points.")

    # Anomaly Detection
    with st.expander("Anomaly Detection"):
        st.markdown("""
        **Purpose**: Detects outliers for risk identification (e.g., extreme weather events).
        **How to Use**: Choose one variable to identify anomalies using Isolation Forest.
        """)
        selected_var_anomaly = st.selectbox("Select Variable for Anomaly Detection", VARIABLES, key="anomaly_var")
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = iso_forest.fit_predict(df[[selected_var_anomaly]])
        fig_anomaly = px.scatter(df, x='time', y=selected_var_anomaly, color='anomaly',
                                 title=f"{selected_var_anomaly.capitalize()} Anomalies", color_continuous_scale='Viridis')
        st.plotly_chart(fig_anomaly, use_container_width=True)

    # Weather Pattern Clustering
    with st.expander("Weather Pattern Clustering"):
        st.markdown("""
        **Purpose**: Groups similar weather conditions for pattern recognition using K-means clustering.
        **How to Use**: Select variables and number of clusters. Analyze cluster characteristics for decision-making (e.g., typical vs. extreme conditions).
        """)
        selected_vars_cluster = st.multiselect("Select Variables for Clustering", VARIABLES, default=['temperature', 'precipitation'], key="cluster_vars")
        n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="cluster_n")
        if selected_vars_cluster and len(selected_vars_cluster) >= 2:
            X = df[selected_vars_cluster].dropna()
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(X)
            fig_cluster = px.scatter(df, x=selected_vars_cluster[0], y=selected_vars_cluster[1], color='cluster',
                                     title="Weather Pattern Clusters", labels={'cluster': 'Cluster'})
            st.plotly_chart(fig_cluster, use_container_width=True)
            st.write("Cluster Centers:")
            st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=selected_vars_cluster))
        else:
            st.warning("Please select at least two variables.")

    # Predictive Modeling with Prophet
    with st.expander("Forecast (Prophet)"):
        st.markdown("""
        **Purpose**: Predicts future weather conditions for planning.
        **How to Use**: Select one variable for forecasting (Historical data only, 30 periods ahead).
        """)
        selected_var_prophet = st.selectbox("Select Variable for Forecast", VARIABLES, key="prophet_var")
        if data_type == "Historical":
            prophet_df = df[['time', selected_var_prophet]].rename(columns={'time': 'ds', selected_var_prophet: 'y'})
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=30, freq='H' if data_type == 'Hourly' else 'D')
            forecast = model.predict(future)
            fig_prophet = go.Figure()
            fig_prophet.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Actual'))
            fig_prophet.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
            fig_prophet.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower CI', line=dict(dash='dash')))
            fig_prophet.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper CI', line=dict(dash='dash')))
            fig_prophet.update_layout(title=f"{selected_var_prophet.capitalize()} Forecast")
            st.plotly_chart(fig_prophet, use_container_width=True)
        else:
            st.info("Forecasting is available for Historical data only.")

    # Correlation Coefficients
    with st.expander("Correlation Coefficients"):
        st.markdown("""
        **Purpose**: Quantifies relationships between variables for predictive modeling.
        **How to Use**: Select multiple variables to compute Pearson correlations and p-values.
        """)
        selected_vars_coeff = st.multiselect("Select Variables for Correlation Coefficients", VARIABLES, default=VARIABLES, key="coeff_vars")
        if selected_vars_coeff and len(selected_vars_coeff) > 1:
            correlations = []
            for i, col1 in enumerate(selected_vars_coeff):
                for col2 in selected_vars_coeff[i+1:]:
                    corr, p_value = pearsonr(df[col1], df[col2])
                    correlations.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Correlation': f"{corr:.3f}",
                        'P-value': f"{p_value:.3e}"
                    })
            st.dataframe(pd.DataFrame(correlations))
        else:
            st.warning("Please select at least two variables.")

    # Decision Matrix
    with st.expander("Decision Matrix"):
        st.markdown("""
        **Purpose**: Provides actionable recommendations based on weather conditions for sectors like agriculture, construction, or aviation.
        **How to Use**: Review recommendations triggered by weather thresholds. Customize thresholds in the Alerts section.
        """)
        decision_matrix = []
        if df['precipitation'].max() > 10:
            decision_matrix.append("Agriculture: Delay planting due to heavy rainfall.")
            decision_matrix.append("Construction: Pause outdoor work due to flood risk.")
        if df['wind_speed'].max() > 50:
            decision_matrix.append("Aviation: Assess wind shear risks for takeoffs/landings.")
            decision_matrix.append("Construction: Secure equipment due to high winds.")
        if df['temperature'].max() > 30:
            decision_matrix.append("Agriculture: Increase irrigation due to high temperatures.")
            decision_matrix.append("Construction: Schedule work in cooler hours.")
        if df['cloud_cover'].mean() > 80:
            decision_matrix.append("Solar Energy: Expect reduced solar output due to high cloud cover.")
        if len(decision_matrix) == 0:
            decision_matrix.append("No critical conditions detected. Proceed with normal operations.")
        for decision in decision_matrix:
            st.write(decision)

    # Data Download and Report
    with st.expander("Download Data and Report"):
        st.markdown("""
        **Purpose**: Exports data and generates a PDF report for offline analysis and decision-making.
        **How to Use**: Download CSV for raw data or PDF for a summarized report with metrics, alerts, and decisions.
        """)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="weather_data.csv",
            mime="text/csv"
        )
        pdf_buffer = generate_pdf_report(df, alerts, decision_matrix)
        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name="weather_report.pdf",
            mime="application/pdf"
        )
else:
    st.info("Select data type, coordinates, and click 'Fetch Weather Data' to load the dashboard.")
