import streamlit as st
import cv2
import time
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from detector import PersonTracker

st.set_page_config(
    page_title="Person Detection Dashboard",
    layout="wide",
    page_icon="🎥",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
    }
    .hero h1 {
        color: white;
        font-weight: 800;
        font-size: 2rem;
        margin: 0;
    }
    .hero p {
        color: rgba(255,255,255,0.85);
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
    }

    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        transition: transform 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6366F1;
    }
    .metric-label {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #F1F5F9;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }
    .metric-icon {
        font-size: 1.4rem;
    }

    .status-live {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(34, 197, 94, 0.15);
        color: #22C55E;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-offline {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22C55E;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.6); }
        70% { box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    .section-title {
        color: #F1F5F9;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 1.2rem 0 0.6rem 0;
        border-left: 4px solid #6366F1;
        padding-left: 0.7rem;
    }

    .video-container {
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

def metric_card(icon, label, value, col):
    col.markdown(f"""
    <div class="metric-card">
        <span class="metric-icon">{icon}</span>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>🎥 Real-Time Person Detection</h1>
    <p>YOLOv8 + ByteTrack &nbsp;•&nbsp; Live tracking, analytics & event logging</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MAIN CONTROL BAR (defines run + reset_btn) ----------------
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 4])
with ctrl_col1:
    run = st.toggle("▶️ Start camera", value=False)
with ctrl_col2:
    reset_btn = st.button("🔄 Reset", use_container_width=True)
with ctrl_col3:
    if run:
        st.markdown('<div class="status-live"><div class="pulse"></div> LIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-offline">⚫ OFFLINE</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR (settings only) ----------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)
    crowd_limit = st.number_input("Crowd alert limit", min_value=1, value=5)
    st.divider()
    st.caption("Model: **YOLOv8n**")
    st.caption("Tracker: **ByteTrack**")

# ---------------- STATE ----------------
if "tracker" not in st.session_state:
    st.session_state.tracker = PersonTracker(conf_threshold=conf_threshold)
if "history" not in st.session_state:
    st.session_state.history = []

if reset_btn:
    st.session_state.tracker.reset()
    st.session_state.history = []
    st.toast("Session reset.", icon="🔄")

st.session_state.tracker.conf_threshold = conf_threshold

# ---------------- TABS ----------------
tab_live, tab_analytics, tab_log = st.tabs(["📹  Live Feed", "📊  Analytics", "📋  Event Log"])

with tab_live:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        frame_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        alert_placeholder = st.empty()

    with col2:
        st.markdown('<div class="section-title">Live Stats</div>', unsafe_allow_html=True)
        m1 = st.empty()
        m2 = st.empty()
        m3 = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access the camera.")
        else:
            prev_time = 0
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame.")
                    break

                annotated_frame, current_count, unique_count = st.session_state.tracker.process_frame(frame)
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time else 0
                prev_time = curr_time

                st.session_state.history.append({"time": datetime.now().strftime("%H:%M:%S"), "count": current_count})
                st.session_state.history = st.session_state.history[-100:]

                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

                with m1.container():
                    metric_card("👥", "People in frame", current_count, st)
                with m2.container():
                    metric_card("🔢", "Unique (session)", unique_count, st)
                with m3.container():
                    metric_card("⚡", "FPS", f"{fps:.1f}", st)

                if current_count >= crowd_limit:
                    alert_placeholder.warning(f"⚠️ Crowd alert — {current_count} people detected (limit: {crowd_limit})")
                else:
                    alert_placeholder.empty()

                run = st.session_state.get("Start camera", run)

            cap.release()
    else:
        st.info("👆 Toggle **Start camera** above to begin detection.")

with tab_analytics:
    st.markdown('<div class="section-title">Count Over Time</div>', unsafe_allow_html=True)
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.line_chart(df.set_index("time")["count"], color="#6366F1")
    else:
        st.caption("No data yet — start the camera to see live analytics.")

    st.markdown('<div class="section-title">Session Summary</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    metric_card("🧍", "Total unique people", len(st.session_state.tracker.seen_ids), c1)
    metric_card("📥", "Total entry events", len(st.session_state.tracker.events), c2)
    max_concurrent = max([h["count"] for h in st.session_state.history], default=0)
    metric_card("📈", "Peak concurrent", max_concurrent, c3)

with tab_log:
    st.markdown('<div class="section-title">Entry Events</div>', unsafe_allow_html=True)
    if st.session_state.tracker.events:
        log_df = pd.DataFrame(st.session_state.tracker.events)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        csv = log_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download log as CSV", csv, "detection_log.csv", "text/csv", use_container_width=False)
    else:
        st.caption("No events logged yet.")