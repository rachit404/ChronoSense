import streamlit as st
import pandas as pd
import os
from pathlib import Path
from main import chrono_sense_pipeline, chat_query
from src.chat_groq_client import ChatGroqClient
from utils.detect_plot import detect_plot_types

# --- PAGE SETUP ---
st.set_page_config(page_title="ChronoSense", page_icon="🐦‍🔥", layout="centered")
st.title("🐦‍🔥 ChronoSense")

# --- DATA DIRECTORY ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File successfully uploaded!")
        
        # Data preview
        with st.expander("👁️ Preview Data", expanded=False):
            st.dataframe(df.head(9))
        
        # Save file
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            df.to_csv(file_path, index=False)
            print(f"[INFO] File saved to: {file_path}")
            
        # col = st.text_input("Column Name: ", value="Close", key="col_name")
        # print("[INFO] Column selected:", col)
        # print("[INFO] Column type:", type(col))

        # file_path = os.path.join(DATA_DIR, uploaded_file.name)
        # df.to_csv(file_path, index=False)
        # print(f"[INFO] File saved to: `{file_path}`")

        # with st.expander("👁️ Preview Data", width=2000):
        #     st.dataframe(df.head(9))
        
        # --- Column name input ---
        col = st.text_input("Column Name:", value="Close", key="col_name")
        if col:
            st.info(f"📊 Selected Column: `{col}`")

            # --- Run pipeline with loader ---
            run_clicked = st.button("🚀 Allow ChronoSense to read csv?")
            result = False
            if run_clicked:
                with st.spinner("⏳ ChronoSense is Reading... Please wait."):
                    result = chrono_sense_pipeline(uploaded_file.name, col)

                if result:
                    st.success("✅ ChronoSense pipeline completed successfully!")
                else:
                    st.error("❌ Pipeline failed or returned False.")
        
        groq_llm = ChatGroqClient()
        # --- CHAT INTERFACE ---
        st.markdown("---")
        st.subheader("💬 Chat with ChronoSense")

        # Initialize session state for chat
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        # --- Clear chat button (top-right) ---
        col1, col2 = st.columns([8, 2])
        with col1:
            st.write("")  # spacer
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                # st.experimental_rerun()

        # Chat input
        user_input = st.chat_input("Ask a question about your data...")

        if user_input:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "message": user_input})

            # Get AI reply
            reply = chat_query(user_input, groq_llm)

            # --- Detect relevant plots from query ---
            plot_files = detect_plot_types(user_input, run_id="007")
            plot_paths = []
            for p in plot_files:
                path = Path(f"visualizations/007/{p}")
                if path.exists():
                    plot_paths.append(str(path))

            # Append AI message (with reply + optional plots)
            st.session_state.chat_history.append({
                "role": "assistant",
                "message": reply,
                "plots": plot_paths
            })


        # --- Display chat history ---
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                with st.chat_message("user", avatar="🎙️"):
                    st.write(chat["message"])

            elif chat["role"] == "assistant":
                with st.chat_message("assistant", avatar="🐦‍🔥"):
                    st.write(chat["message"])

                    # If assistant has plots, display them
                    if "plots" in chat and chat["plots"]:
                        st.markdown("**📊 Related Visualizations:**")
                        cols = st.columns(len(chat["plots"]))
                        for i, plot_path in enumerate(chat["plots"]):
                            with cols[i]:
                                st.image(plot_path, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")
else:
    st.warning("Please upload a CSV file to continue.")

