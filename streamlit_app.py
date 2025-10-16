import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}/api/query"

st.set_page_config(page_title="Knowledge-Base Search Engine", page_icon="📚")
st.title("📚 Knowledge-Base Search Engine (Local Free)")

query = st.text_input("🔍 Enter your question:")
top_k = st.number_input("📄 Number of context chunks",
                        value=4, min_value=1, max_value=10)

if st.button("Ask"):
    if not query.strip():
        st.warning("⚠️ Please enter a question before asking.")
    else:
        with st.spinner("🤖 Generating answer..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": query, "top_k": top_k},
                    timeout=60
                )
                if response.ok:
                    data = response.json()
                    st.success("✅ Answer generated successfully!")

                    st.subheader("🧠 Answer")
                    st.write(data.get("answer", "No answer returned."))

                    st.subheader("📚 Source Documents")
                    for i, s in enumerate(data.get("sources", []), 1):
                        st.markdown(
                            f"**Source {i}:** {s['page_content'][:500]}...")
                else:
                    st.error(f"❌ Server error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(
                    "🚫 Cannot connect to backend API. Make sure `app.py` is running.")
            except requests.exceptions.Timeout:
                st.error("⏰ The request timed out. Model might be slow on CPU.")
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {e}")
