import streamlit as st
import requests

st.set_page_config(page_title="TurboMind", layout="wide")

st.title("🚀 TurboMind AI Assistant (TurboQuant Simulation)")

# 🔥 TurboQuant Control
bits = st.slider("Compression Level (Bits)", 2, 8, 4)

query = st.text_input("Ask a medical question:")

if st.button("Send"):
    try:
        resp = requests.post(
            "http://127.0.0.1:8000/chat",
            params={"query": query, "bits": bits},
            timeout=120
        )
        resp.raise_for_status()
        res = resp.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    # 🤖 Response
    st.subheader("🤖 Response")
    st.write(res["response"])

    # 📊 TurboQuant Visualization
    st.subheader("📊 TurboQuant Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Original Context", res["original_length"])
    col2.metric("Compressed Context", res["compressed_length"])
    col3.metric("Compression Ratio", res["compression_ratio"])

    # ⚡ Performance
    st.subheader("⚡ Performance")
    st.write(f"Memory Usage: {res['memory_usage']}")
    st.write(f"Latency: {res['latency']} sec")
    st.write(f"Bits Used: {res['bits']}")

    # 📉 Chart
    st.subheader("📉 Compression Comparison")

    chart_data = {
        "Before": res["original_length"],
        "After": res["compressed_length"]
    }

    st.bar_chart(chart_data)
