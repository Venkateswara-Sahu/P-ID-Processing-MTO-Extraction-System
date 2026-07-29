"""
Streamlit Dashboard for P&ID MTO Extraction Pipeline.

Interactive web interface for uploading P&IDs, visualizing detections,
exploring the entity graph, and downloading MTO spreadsheets.
"""

import io
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

# Page configuration
st.set_page_config(
    page_title="P&ID MTO Extractor — AI Pipeline",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-card h3 {
        color: #667eea;
        font-size: 2rem;
        margin: 0;
    }
    .metric-card p {
        color: rgba(255,255,255,0.7);
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-pass {
        background: rgba(76, 175, 80, 0.2);
        color: #4CAF50;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    .status-fail {
        background: rgba(244, 67, 54, 0.2);
        color: #F44336;
        border: 1px solid rgba(244, 67, 54, 0.3);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: #1a1a2e;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 8px 16px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit application."""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏗️ P&ID Document Intelligence</h1>
        <p>AI-Powered Material Take-Off Extraction Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Settings")

        confidence_threshold = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=0.95,
            value=0.5,
            step=0.05,
            help="Minimum confidence for symbol detection",
        )

        run_llm = st.checkbox(
            "Run LLM Validation",
            value=True,
            help="Use Groq Llama 3.3 for AI-powered validation",
        )

        project_name = st.text_input(
            "Project Name",
            value="P&ID Extraction Demo",
            help="Name for the MTO report header",
        )

        st.markdown("---")
        st.markdown("### 📊 Pipeline Architecture")
        st.markdown("""
        1. 📄 **Preprocess** — Enhance & upscale
        2. 🎯 **Detect** — YOLOv8 symbols
        3. 📝 **Extract** — Tesseract OCR (CC-guided)
        4. 🔗 **Map** — Entity graph (NetworkX)
        5. 🤖 **Validate** — LLM + rules
        6. 📊 **Generate** — MTO Excel
        """)

        st.markdown("---")
        st.markdown(
            "Built with YOLOv8, Tesseract OCR, LangGraph, Groq, NetworkX",
            help="Free & open-source stack",
        )

    # Main content
    uploaded_file = st.file_uploader(
        "Upload P&ID Drawing",
        type=["png", "jpg", "jpeg", "pdf", "tiff", "bmp"],
        help="Upload a P&ID drawing (image or PDF) for MTO extraction",
    )

    if uploaded_file is not None:
        # Display uploaded image
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset for later use

        # Convert to numpy array
        if uploaded_file.type == "application/pdf":
            st.info("📄 PDF uploaded — will process first page")
            # Save temporarily for PDF processing
            temp_path = Path(settings.output_dir) / "temp_upload.pdf"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            input_source = str(temp_path)
            # Show PDF info
            st.markdown(f"**File:** {uploaded_file.name} | **Size:** {len(file_bytes)/1024:.1f} KB")
        else:
            # Image file
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Failed to load image")
                return

            input_source = image

            # Show preview
            preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(preview, caption=f"📄 {uploaded_file.name} ({image.shape[1]}×{image.shape[0]}px)", width="stretch")

        # Process button
        if st.button("🚀 Extract MTO", type="primary", width="stretch"):
            _run_pipeline(input_source, project_name, run_llm, confidence_threshold)

        # Display results from session state (persists across reruns)
        if "pipeline_result" in st.session_state:
            _display_results(st.session_state["pipeline_result"])

    else:
        # Clear results when no file is uploaded
        st.session_state.pop("pipeline_result", None)
        # Welcome state
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯</h3>
                <p><b>Symbol Detection</b><br>YOLOv8 detects valves, instruments, equipment</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📝</h3>
                <p><b>OCR Extraction</b><br>PaddleOCR reads tags, line numbers, specs</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🤖</h3>
                <p><b>AI Validation</b><br>LangGraph + Groq verify engineering rules</p>
            </div>
            """, unsafe_allow_html=True)


def _run_pipeline(input_source, project_name, run_llm, confidence_threshold):
    """Run the extraction pipeline with progress display."""
    from src.pipeline import PIDPipeline

    progress_bar = st.progress(0, text="Initializing pipeline...")

    try:
        pipe = PIDPipeline()

        progress_bar.progress(10, text="🔧 Preprocessing image...")
        time.sleep(0.3)

        progress_bar.progress(20, text="🎯 Detecting symbols (YOLOv8)...")

        # Run pipeline
        if isinstance(input_source, str):
            result = pipe.process(
                input_path=input_source,
                project_name=project_name,
                run_llm_validation=run_llm,
                generate_mto=True,
            )
        else:
            result = pipe.process_image_array(
                image=input_source,
                project_name=project_name,
                run_llm_validation=run_llm,
                generate_mto=True,
            )

        progress_bar.progress(100, text="✅ Pipeline complete!")
        time.sleep(0.5)
        progress_bar.empty()

        # Store results in session state so they survive reruns
        st.session_state["pipeline_result"] = result

    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Pipeline failed: {str(e)}")
        st.exception(e)


def _display_results(result):
    """Display pipeline results in a beautiful dashboard."""

    # Summary metrics
    st.markdown("### 📊 Extraction Summary")
    summary = result.to_summary()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Entities", summary["total_entities"])
    with col2:
        st.metric("Detections", summary["total_detections"])
    with col3:
        st.metric("Text Regions", summary["total_text_regions"])
    with col4:
        st.metric("Graph Nodes", summary["graph"].get("total_nodes", 0))
    with col5:
        st.metric("Time (sec)", f"{summary['processing_time_seconds']:.1f}")

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Detections", "📋 MTO Table", "🔗 Graph", "✅ Validation", "📥 Download"
    ])

    with tab1:
        _display_detection_view(result)

    with tab2:
        _display_mto_table(result)

    with tab3:
        _display_graph_view(result)

    with tab4:
        _display_validation(result)

    with tab5:
        _display_download(result)


def _display_detection_view(result):
    """Show annotated P&ID with detection overlays."""
    st.markdown("#### Annotated P&ID — Detected Symbols & Text")

    if result.annotated_image is not None:
        annotated_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Green=Valves | Orange=Instruments | Red=Equipment | Blue=Text", width="stretch")

    # Detection breakdown
    if result.detections:
        from src.detection.postprocessor import DetectionPostProcessor
        summary = DetectionPostProcessor.get_detection_summary(result.detections)

        st.markdown("#### Detection Breakdown")
        for class_name, info in summary.items():
            st.markdown(
                f"- **{class_name}**: {info['count']} detected "
                f"(avg confidence: {info['avg_confidence']:.2%})"
            )


def _display_mto_table(result):
    """Show the extracted MTO data as an interactive table."""
    st.markdown("#### Material Take-Off (MTO) Data")

    if not result.entities:
        st.info("No entities extracted")
        return

    import pandas as pd

    # Convert entities to DataFrame
    rows = []
    for i, entity in enumerate(result.entities, 1):
        rows.append({
            "Item": i,
            "Tag": entity.tag_id or "—",
            "Description": entity.description or "Unknown",
            "Type": (entity.entity_type or "unknown").title(),
            "Category": (entity.component_category or "other").title(),
            "Size": entity.size or "—",
            "Spec": entity.specification or "—",
            "Line No.": entity.line_number or "—",
            "Confidence": f"{entity.confidence:.0%}",
        })

    df = pd.DataFrame(rows)

    # Style the dataframe
    st.dataframe(
        df,
        width="stretch",
        height=400,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                format="%.0f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

    st.caption(f"Total: {len(rows)} components extracted")


def _display_graph_view(result):
    """Show the entity relationship graph."""
    st.markdown("#### P&ID Entity Relationship Graph")

    if result.graph is None or result.graph.number_of_nodes() == 0:
        st.info("No graph data available")
        return

    # Graph summary
    graph_summary = result.graph_summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nodes", graph_summary.get("total_nodes", 0))
    with col2:
        st.metric("Edges", graph_summary.get("total_edges", 0))
    with col3:
        st.metric("Components", graph_summary.get("connected_components", 0))

    # Entity type breakdown
    type_counts = graph_summary.get("entity_types", {})
    if type_counts:
        st.markdown("**Entity Types:**")
        for etype, count in type_counts.items():
            st.markdown(f"- {etype}: **{count}**")

    # Interactive graph visualization using pyvis
    try:
        from pyvis.network import Network
        import tempfile

        net = Network(height="500px", width="100%", bgcolor="#1a1a2e", font_color="white")
        net.barnes_hut()

        color_map = {
            "valves": "#4CAF50",
            "instruments": "#FF9800",
            "equipment": "#F44336",
            "piping": "#2196F3",
            "other": "#9E9E9E",
        }

        for node_id, data in result.graph.nodes(data=True):
            category = data.get("category", "other")
            label = str(data.get("tag_id") or data.get("description", node_id))
            color = color_map.get(category, "#9E9E9E")
            net.add_node(node_id, label=label, color=color, size=20)

        for source, target, data in result.graph.edges(data=True):
            conn_type = data.get("connection_type", "piping")
            color = "#00BCD4" if conn_type == "signal" else "#607D8B"
            net.add_edge(source, target, color=color)

        # Save and display
        temp_dir = Path(settings.output_dir) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        graph_html = str(temp_dir / "graph.html")
        net.save_graph(graph_html)

        with open(graph_html, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Render the PyVis graph in an iframe via components.v1.html
        # (st.html wraps in a div which breaks full HTML documents)
        import streamlit.components.v1 as components
        components.html(html_content, height=520, scrolling=True)

    except ImportError:
        st.warning("Install pyvis for interactive graph visualization: `pip install pyvis`")
        st.json(graph_summary)


def _display_validation(result):
    """Show validation results."""
    st.markdown("#### Validation Results")

    # Rule-based validation
    rule_report = result.rule_report
    if rule_report:
        summary = rule_report.get("summary", {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pass Rate", summary.get("pass_rate", "N/A"))
        with col2:
            st.metric("Errors", summary.get("errors", 0))
        with col3:
            st.metric("Warnings", summary.get("warnings", 0))

        issues = rule_report.get("issues", [])
        if issues:
            st.markdown("**Issues Found:**")
            for issue in issues:
                severity = issue.get("severity", "info")
                icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "⚪")
                st.markdown(
                    f"{icon} **[{issue.get('rule_id', '')}]** {issue.get('message', '')}"
                )
                if issue.get("suggestion"):
                    st.caption(f"   💡 {issue['suggestion']}")

    # LLM validation
    if result.llm_report and result.llm_report.get("report"):
        st.markdown("---")
        st.markdown("#### 🤖 AI Validation Report (Llama 3.3 70B)")
        st.markdown(result.llm_report["report"])

    # Confidence summary
    if result.confidence_summary:
        st.markdown("---")
        st.markdown("#### Confidence Summary")
        conf = result.confidence_summary

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Confidence", f"{conf.get('avg_confidence', 0):.1%}")
        with col2:
            st.metric("High Confidence", conf.get("high_confidence", 0))
        with col3:
            st.metric("Needs Review", conf.get("needs_review", 0))


def _display_download(result):
    """Show download options."""
    st.markdown("#### 📥 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        # MTO Excel download
        if result.mto_path and Path(result.mto_path).exists():
            with open(result.mto_path, "rb") as f:
                st.download_button(
                    "📊 Download MTO Excel",
                    data=f.read(),
                    file_name=Path(result.mto_path).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    width="stretch",

                )
        else:
            st.info("MTO file not generated yet")

    with col2:
        # Annotated image download
        if result.annotated_image is not None:
            _, buffer = cv2.imencode(".png", result.annotated_image)
            st.download_button(
                "🖼️ Download Annotated P&ID",
                data=buffer.tobytes(),
                file_name="annotated_pid.png",
                mime="image/png",
                width="stretch",
            )

    # JSON export
    st.markdown("---")
    summary_json = json.dumps(result.to_summary(), indent=2, default=str)
    st.download_button(
        "📋 Download JSON Summary",
        data=summary_json,
        file_name="extraction_summary.json",
        mime="application/json",
        width="stretch",
    )


if __name__ == "__main__":
    main()
