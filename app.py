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
# 1️⃣ LOAD DATASET FROM GITHUB
# -------------------------------

DATA_URL = "https://raw.githubusercontent.com/Tanish-analyst/ecommerce-analytics-agent/main/ecommerce_sales_dataset.xlsx"

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_URL)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


df = load_data()
st.success("Dataset loaded successfully!")
st.dataframe(df.head())

# Make global df accessible
global_df = df

# -------------------------------
# 2️⃣ SAFE EXECUTION FUNCTION
# -------------------------------

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

# -------------------------------
# 3️⃣ TOOL: sales_query
# -------------------------------

@tool
def sales_query(code: str) -> str:
    """Run Pandas code against the ecommerce dataset."""
    return execute_df_query(code)

tools = [sales_query]

# -------------------------------
# 4️⃣ LLM SETUP
# -------------------------------

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

llm_basic = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, tools=tools)

# -------------------------------
# 5️⃣ LANGGRAPH STATE
# -------------------------------

class AgentState(TypedDict):
    messages: List

# -------------------------------
# 6️⃣ MODEL NODE
# -------------------------------

def model_node(state: AgentState) -> AgentState:

    messages = state["messages"]
    last_message = messages[-1]

    user_query = last_message.content

    # Step 1: Ask LLM to generate Pandas code
    prompt = f"""
You are a senior data analyst. Generate ONLY Pandas code to answer the user's question.

USER QUESTION:
"{user_query}"

DATAFRAME NAME: df  

SCHEMA:
- order_id
- order_date
- customer_id
- product_id
- product_name
- category
- quantity
- price
- discount
- revenue
- cost
- profit
- payment_method
- city
- stock_left

RULES:
1. Use df directly.
2. Use df['revenue'] directly (it already includes discount).
3. Filter dates with df['order_date'].dt.year / month.
4. Output ONLY valid Pandas code.
5. No markdown. No explanations. Only code.
"""

    response = llm_basic.invoke(prompt)
    code = response.content.strip()

    # Step 2: Execute code
    result = sales_query.invoke({"code": code})

    # Step 3: Ask LLM to interpret result
    interpret_prompt = f"""
Question: {user_query}

Pandas Code:
{code}

Execution Result:
{result}

Give a clear, simple final answer.
"""

    final_response = llm_basic.invoke(interpret_prompt)

    state["messages"].append(AIMessage(content=final_response.content))
    return state

# -------------------------------
# 7️⃣ ROUTER
# -------------------------------

def router(state: AgentState) -> str:
    return END

# -------------------------------
# 8️⃣ BUILD WORKFLOW
# -------------------------------

workflow = StateGraph(AgentState)
workflow.add_node("model", model_node)
workflow.set_entry_point("model")
workflow.add_edge("model", END)

app = workflow.compile()

# -------------------------------
# 9️⃣ STREAMLIT UI
# -------------------------------

st.title("📊 Ecommerce Analytics AI Agent")
st.write("Ask any analytics question about your ecommerce dataset.")

user_input = st.text_input("Enter your question:")

if user_input:
    with st.spinner("Thinking..."):
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        result = app.invoke(initial_state)

        # Get final AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                st.subheader("💡 Answer")
                st.write(msg.content)
                break

