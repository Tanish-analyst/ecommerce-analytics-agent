# import streamlit as st
# import pandas as pd
# import requests
# from langchain_groq import ChatGroq
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.tools import tool
# from langgraph.graph import StateGraph, END
# from typing import TypedDict, List
# import os

# # -------------------------------
# # 1️⃣ LOAD DATASET FROM GITHUB
# # -------------------------------

# DATA_URL = "https://raw.githubusercontent.com/Tanish-analyst/ecommerce-analytics-agent/main/ecommerce_sales_dataset.xlsx"

# langchain_tracing = st.secrets["LANGCHAIN_TRACING_V2"] 
# langsmith_project = st.secrets["LANGSMITH_PROJECT"] 
# langsmith_api_key = st.secrets["LANGSMITH_API_KEY"]


# @st.cache_data
# def load_data():
#     df = pd.read_excel(DATA_URL)
#     df["order_date"] = pd.to_datetime(df["order_date"])
#     return df


# df = load_data()
# st.success("Dataset loaded successfully!")
# st.dataframe(df.head())

# # Make global df accessible
# global_df = df

# def clean_llm_code(text: str) -> str:
#     """Extract only raw Python code from LLM output."""
#     import re
    
#     # Remove markdown ``` blocks
#     text = text.replace("```python", "").replace("```", "")
    
#     # Remove "Here is the code:", "The code is:", etc.
#     patterns = [
#         r"Here is.*?:",
#         r"The code is.*?:",
#         r"Use this code.*?:",
#         r"Corrected code.*?:",
#         r"Improved version.*?:",
#         r"Try this.*?:"
#     ]
#     for p in patterns:
#         text = re.sub(p, "", text, flags=re.IGNORECASE)
    
#     # Remove JSON wrapper {"code": "..."}
#     if text.strip().startswith("{"):
#         try:
#             import json
#             parsed = json.loads(text)
#             if "code" in parsed:
#                 return parsed["code"].strip()
#         except:
#             pass
    
#     return text.strip()

# # -------------------------------
# # 2️⃣ SAFE EXECUTION FUNCTION
# # -------------------------------

# def execute_df_query(code: str):
#     """Executes Pandas code safely on df."""
#     try:
#         local_vars = {"df": global_df, "pd": pd}
#         exec(f"__result__ = {code}", {"__builtins__": {}, "pd": pd}, local_vars)
#         result = local_vars.get("__result__")

#         if isinstance(result, pd.DataFrame):
#             return result.head().to_string()
#         elif isinstance(result, pd.Series):
#             return result.to_string()
#         else:
#             return str(result)

#     except Exception as e:
#         return f"Error executing code → {e}"

# # -------------------------------
# # 3️⃣ TOOL: sales_query
# # -------------------------------

# @tool
# def sales_query(code: str) -> str:
#     """Run Pandas code against the ecommerce dataset."""
#     return execute_df_query(code)

# tools = [sales_query]

# # -------------------------------
# # 4️⃣ LLM SETUP
# # -------------------------------

# os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# llm_basic = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# llm_with_tools = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, tools=tools)

# # -------------------------------
# # 5️⃣ LANGGRAPH STATE
# # -------------------------------

# class AgentState(TypedDict):
#     messages: List

# # -------------------------------
# # 6️⃣ MODEL NODE
# # -------------------------------

# def model_node(state: AgentState) -> AgentState:

#     messages = state["messages"]
#     last_message = messages[-1]

#     user_query = last_message.content

#     # Step 1: Ask LLM to generate Pandas code
#     prompt = f"""
# You are a senior data analyst. Your job is to write ONLY valid Pandas code that answers the user's question.

# DATAFRAME NAME → df  

# DATASET SCHEMA:
# - order_id (int): unique order ID
# - order_date (datetime): purchase date
# - customer_id (int)
# - product_id (int)
# - product_name (str)
# - category (str): examples include "Electronics", "Fashion/Home", "Clothing", "Home", etc.
# - quantity (int): quantity purchased
# - price (float): price per unit
# - discount (int): discount percent
# - revenue (float): final revenue AFTER discount (already calculated)
# - cost (float)
# - profit (float): revenue – cost
# - payment_method (str): e.g., "Credit Card", "UPI", "COD", etc.
# - city (str): e.g., Delhi, Mumbai, Bangalore, Chennai
# - stock_left (int)

# VERY IMPORTANT RULES:
# 1. Use df exactly as it is.
# 2. Use df['revenue'] directly — it already includes discount.
# 3. Use df['profit'] directly.
# 4. Filter dates using:
#    - df['order_date'].dt.year
#    - df['order_date'].dt.month
# 5. Output MUST be ONLY raw Pandas code.
# 6. NO explanations, NO markdown, NO text.
# 7. Do NOT wrap the result in JSON or in any dict-like structure such as code: "...".
# 8. Multi-line Python code is allowed.
# 9. The output should be directly executable with exec().

# USER QUESTION:
# "{user_query}"

# """

#     response = llm_basic.invoke(prompt)
#     code = clean_llm_code(response.content.strip())

#     # Step 2: Execute code
#     result = sales_query.invoke({"code": code})

#     # Step 3: Ask LLM to interpret result
#     interpret_prompt = f"""
# Question: {user_query}

# Pandas Code:
# {code}

# Execution Result:
# {result}

# Give a clear, simple final answer.
# """

#     final_response = llm_basic.invoke(interpret_prompt)

#     state["messages"].append(AIMessage(content=final_response.content))
#     return state

# # -------------------------------
# # 7️⃣ ROUTER
# # -------------------------------

# def router(state: AgentState) -> str:
#     return END

# # -------------------------------
# # 8️⃣ BUILD WORKFLOW
# # -------------------------------

# workflow = StateGraph(AgentState)
# workflow.add_node("model", model_node)
# workflow.set_entry_point("model")
# workflow.add_edge("model", END)

# app = workflow.compile()

# # -------------------------------
# # 9️⃣ STREAMLIT UI
# # -------------------------------

# st.title("📊 Ecommerce Analytics AI Agent")
# st.write("Ask any analytics question about your ecommerce dataset.")

# user_input = st.text_input("Enter your question:")

# if user_input:
#     with st.spinner("Thinking..."):
#         initial_state = {"messages": [HumanMessage(content=user_input)]}
#         result = app.invoke(initial_state)

#         # Get final AI message
#         for msg in reversed(result["messages"]):
#             if isinstance(msg, AIMessage):
#                 st.subheader("💡 Answer")
#                 st.write(msg.content)
#                 break




import streamlit as st
import pandas as pd
import requests
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import os

# -------------------------------
# 0️⃣ STREAMLIT PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Ecommerce Analytics AI Agent",
    page_icon="📊",
    layout="wide",
)

# -------------------------------
# 🌈 GLOBAL STYLING (CUSTOM CSS)
# -------------------------------

st.markdown(
    """
    <style>
    /* Global background */
    .stApp {
        background: radial-gradient(circle at top left, #f9fafb 0, #eef2ff 40%, #f9fafb 100%);
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Center the main block a bit more and widen it */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title styling */
    .app-header {
        padding: 1.25rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #1d4ed8, #4f46e5);
        color: white;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.35);
        margin-bottom: 1.5rem;
    }

    .app-header-title {
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .app-header-subtitle {
        font-size: 0.95rem;
        opacity: 0.9;
    }

    /* Info cards */
    .info-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
        backdrop-filter: blur(12px);
    }

    .info-card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.4rem;
    }

    .info-card-body {
        font-size: 0.85rem;
        color: #475569;
    }

    /* Section title */
    .section-label {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }

    /* Answer card */
    .answer-card {
        margin-top: 1rem;
        background: #ffffff;
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.16);
    }

    .answer-card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    .answer-card-body {
        font-size: 0.95rem;
        color: #111827;
        line-height: 1.6;
        white-space: pre-wrap;
    }

    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 999px !important;
        border: 1px solid rgba(148, 163, 184, 0.8) !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: none !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 1px rgba(79, 70, 229, 0.5) !important;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 999px;
        padding: 0.4rem 1.4rem;
        font-weight: 600;
        border: none;
        background: linear-gradient(120deg, #4f46e5, #2563eb);
        color: white;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.4);
    }

    .stButton > button:hover {
        background: linear-gradient(120deg, #4338ca, #1d4ed8);
        box-shadow: 0 14px 28px rgba(30, 64, 175, 0.55);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        font-weight: 600;
    }

    /* Sidebar tweaks */
    [data-testid="stSidebar"] {
        background: #0f172a;
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# 1️⃣ LOAD DATASET FROM GITHUB
# -------------------------------

DATA_URL = "https://raw.githubusercontent.com/Tanish-analyst/ecommerce-analytics-agent/main/ecommerce_sales_dataset.xlsx"

langchain_tracing = st.secrets["LANGCHAIN_TRACING_V2"]
langsmith_project = st.secrets["LANGSMITH_PROJECT"]
langsmith_api_key = st.secrets["LANGSMITH_API_KEY"]


@st.cache_data
def load_data():
    df = pd.read_excel(DATA_URL)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


df = load_data()
global_df = df  # Make global df accessible

# -------------------------------
# 🧊 SIDEBAR (DATASET QUICK INFO)
# -------------------------------

with st.sidebar:
    st.markdown("### 📁 Dataset Overview")
    rows, cols = df.shape
    st.metric("Total Rows", f"{rows:,}")
    st.metric("Total Columns", cols)

    if "order_date" in df.columns:
        try:
            min_date = df["order_date"].min().date()
            max_date = df["order_date"].max().date()
            st.caption(f"🗓 Date Range: {min_date} → {max_date}")
        except Exception:
            pass

    st.markdown("---")
    st.markdown("#### 🔍 Common Questions")
    st.caption(
        "- Total revenue in 2024\n"
        "- Top 5 most profitable cities\n"
        "- Which category has the highest average discount?\n"
        "- Monthly revenue trend in 2023"
    )
    st.markdown("---")
    st.caption("Built for internal analytics & client demos.")

# -------------------------------
# 🏷 HEADER
# -------------------------------

st.markdown(
    """
    <div class="app-header">
        <div class="app-header-title">📊 Ecommerce Analytics AI Agent</div>
        <div class="app-header-subtitle">
            Ask natural-language questions about your ecommerce sales data – the AI will write Pandas, run it on the dataset, and explain the result.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# 🔍 TOP SECTION (STATS + HOW TO USE)
# -------------------------------

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card-title">Dataset Loaded Successfully ✅</div>
        <div class="info-card-body">
            Your ecommerce dataset is connected and ready.  
            Ask questions like:
            <ul>
              <li>“What was the total revenue in 2024?”</li>
              <li>“Which product generated the highest profit?”</li>
              <li>“Top 5 cities by revenue in 2023?”</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-card-title">How It Works ⚙️</div>
        <div class="info-card-body">
            1. You type a question.<br>
            2. The AI generates Pandas code using <code>df</code>.<br>
            3. The code executes on the live dataset.<br>
            4. The AI explains the result in plain English.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 📄 DATASET PREVIEW (OPTIONAL)
# -------------------------------

st.markdown('<div class="section-label">Dataset Preview</div>', unsafe_allow_html=True)
with st.expander("🔎 Show first 10 rows of the dataset"):
    st.dataframe(df.head(10), use_container_width=True)

# -------------------------------
# ⚙️ LLM + TOOLING LOGIC
# -------------------------------

def clean_llm_code(text: str) -> str:
    """Extract only raw Python code from LLM output."""
    import re

    # Remove markdown ``` blocks
    text = text.replace("```python", "").replace("```", "")

    # Remove "Here is the code:", "The code is:", etc.
    patterns = [
        r"Here is.*?:",
        r"The code is.*?:",
        r"Use this code.*?:",
        r"Corrected code.*?:",
        r"Improved version.*?:",
        r"Try this.*?:",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    # Remove JSON wrapper {"code": "..."}
    if text.strip().startswith("{"):
        try:
            import json

            parsed = json.loads(text)
            if "code" in parsed:
                return parsed["code"].strip()
        except Exception:
            pass

    return text.strip()


def execute_df_query(code: str):
    """Executes Pandas code safely on df."""
    try:
        local_vars = {"df": global_df, "pd": pd}
        exec(f"__result__ = {code}", {"__builtins__": {}, "pd": pd}, local_vars)
        result = local_vars.get("__result__")

        if isinstance(result, pd.DataFrame):
            return result.head().to_string()
        elif isinstance(result, pd.Series):
            return result.to_string()
        else:
            return str(result)

    except Exception as e:
        return f"Error executing code → {e}"


@tool
def sales_query(code: str) -> str:
    """Run Pandas code against the ecommerce dataset."""
    return execute_df_query(code)


tools = [sales_query]

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

llm_basic = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, tools=tools)


class AgentState(TypedDict):
    messages: List


def model_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    user_query = last_message.content

    # Step 1: Ask LLM to generate Pandas code
    prompt = f"""
You are a senior data analyst. Your job is to write ONLY valid Pandas code that answers the user's question.

DATAFRAME NAME → df  

DATASET SCHEMA:
- order_id (int): unique order ID
- order_date (datetime): purchase date
- customer_id (int)
- product_id (int)
- product_name (str)
- category (str): examples include "Electronics", "Fashion/Home", "Clothing", "Home", etc.
- quantity (int): quantity purchased
- price (float): price per unit
- discount (int): discount percent
- revenue (float): final revenue AFTER discount (already calculated)
- cost (float)
- profit (float): revenue – cost
- payment_method (str): e.g., "Credit Card", "UPI", "COD", etc.
- city (str): e.g., Delhi, Mumbai, Bangalore, Chennai
- stock_left (int)

VERY IMPORTANT RULES:
1. Use df exactly as it is.
2. Use df['revenue'] directly — it already includes discount.
3. Use df['profit'] directly.
4. Filter dates using:
   - df['order_date'].dt.year
   - df['order_date'].dt.month
5. Output MUST be ONLY raw Pandas code.
6. NO explanations, NO markdown, NO text.
7. Do NOT wrap the result in JSON or in any dict-like structure such as code: "...".
8. Multi-line Python code is allowed.
9. The output should be directly executable with exec().

USER QUESTION:
"{user_query}"
"""
    response = llm_basic.invoke(prompt)
    code = clean_llm_code(response.content.strip())

    # Step 2: Execute code
    result = sales_query.invoke({"code": code})

    # Step 3: Ask LLM to interpret result
    interpret_prompt = f"""
Question: {user_query}

Pandas Code:
{code}

Execution Result:
{result}

Give a clear, concise final answer that a business stakeholder can easily understand.
If there is an error, explain what went wrong and how to fix the question or logic.
"""

    final_response = llm_basic.invoke(interpret_prompt)

    state["messages"].append(AIMessage(content=final_response.content))
    return state


def router(state: AgentState) -> str:
    return END


workflow = StateGraph(AgentState)
workflow.add_node("model", model_node)
workflow.set_entry_point("model")
workflow.add_edge("model", END)

app = workflow.compile()

# -------------------------------
# 💬 MAIN Q&A UI
# -------------------------------

st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)

query_col, button_col = st.columns([3, 1])

with query_col:
    user_input = st.text_input(
        " ",
        placeholder="Example: What was the total revenue in 2024?",
        label_visibility="collapsed",
    )

with button_col:
    ask_clicked = st.button("Ask AI")

# Trigger on button click or when user presses enter
if (ask_clicked or user_input) and user_input.strip():
    with st.spinner("Analyzing your question and running Pandas on the dataset..."):
        initial_state = {"messages": [HumanMessage(content=user_input.strip())]}
        result = app.invoke(initial_state)

        final_answer = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                final_answer = msg.content
                break

    if final_answer is not None:
        st.markdown(
            """
            <div class="section-label">Result</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="answer-card">
                <div class="answer-card-title">💡 AI Answer</div>
                <div class="answer-card-body">{final_answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
