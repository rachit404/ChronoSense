import streamlit as st
from app.data_handler import load_dataset, validate_dataset, preprocess_dataset
from app.rag_pipeline import create_rag_chain
from app.analytics import handle_analytical_query

# Streamlit page config
st.set_page_config(page_title="ChronoSense", layout="wide")
st.title("ChronoSense: Chat with Your Time-Series Data")

# -------------------------
# Session State Initialization
# -------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "datetime_col" not in st.session_state:
    st.session_state.datetime_col = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# Step 1: Dataset Upload
# -------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV/Excel time-series dataset", type=["csv", "xls", "xlsx"]
)

if uploaded_file and st.session_state.df is None:
    try:
        # Load and preprocess dataset
        df = load_dataset(uploaded_file)
        datetime_col, df = validate_dataset(df)
        df = preprocess_dataset(df, datetime_col)
        st.session_state.df = df
        st.session_state.datetime_col = datetime_col

        st.success("✅ Dataset uploaded and processed successfully")
        st.subheader("Preview of Dataset")
        st.dataframe(df.head())

        st.subheader("Dataset Summary")
        st.write(df.describe())

        # Initialize RAG chain
        st.session_state.qa_chain = create_rag_chain()
        st.info("RAG pipeline initialized. You can now ask questions about your dataset.")

    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# -------------------------
# Step 2: Chat Interface
# -------------------------
if st.session_state.df is not None:
    query = st.text_input("Ask a question about your dataset:", "")
    if st.button("Send") and query:
        # Handle analytical queries or fallback to RAG LLM
        response = handle_analytical_query(
            st.session_state.df,
            st.session_state.datetime_col,
            query,
            qa_chain=st.session_state.qa_chain
        )

        # Append to chat history
        st.session_state.chat_history.append({"user": query, "bot": response})

# -------------------------
# Step 3: Display Chat History
# -------------------------
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    bot_resp = chat["bot"]

    if bot_resp["type"] == "plot":
        st.markdown(f"**ChronoSense:** {bot_resp['description']}")
        st.image(bot_resp["buffer"])
    elif bot_resp["type"] == "forecast":
        st.markdown(f"**ChronoSense:** {bot_resp['description']}")
        st.dataframe(bot_resp["data"].head())
    elif bot_resp["type"] == "stats":
        st.markdown(f"**ChronoSense:** {bot_resp['description']}")
        st.json(bot_resp["data"])
    elif bot_resp["type"] == "llm":
        st.markdown(f"**ChronoSense:** {bot_resp['answer']}")
    else:
        st.markdown(f"**ChronoSense:** {bot_resp.get('message', 'Unable to process query.')}")
