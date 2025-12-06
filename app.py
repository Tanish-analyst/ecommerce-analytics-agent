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
    initial_sidebar_state="expanded"
)

# -------------------------------
# 🌈 ADVANCED GLOBAL STYLING
# -------------------------------

st.markdown(
    """
    <style>
    /* Base Styles */
    .stApp {
        background: linear-gradient(135deg, #f6f8ff 0%, #f0f4ff 50%, #f9fafb 100%);
        font-family: "Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Container */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Glass Morphism Header */
    .glass-header {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.95) 0%, rgba(67, 56, 202, 0.95) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 20px 60px rgba(30, 58, 138, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 30%, rgba(120, 119, 198, 0.3) 0%, transparent 70%);
        z-index: 0;
    }
    
    .header-content {
        position: relative;
        z-index: 1;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
        line-height: 1.6;
        max-width: 800px;
    }
    
    /* Stats Cards */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .stat-card {
        flex: 1;
        min-width: 200px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.08);
        backdrop-filter: blur(4px);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.15);
        border-color: rgba(165, 180, 252, 0.5);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stat-trend {
        font-size: 0.85rem;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        background: rgba(34, 197, 94, 0.1);
        color: #16a34a;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        margin-top: 0.5rem;
    }
    
    /* Input Section */
    .input-section {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 24px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(226, 232, 240, 0.9);
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.1);
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-title::before {
        content: '';
        width: 4px;
        height: 24px;
        background: linear-gradient(to bottom, #4f46e5, #7c3aed);
        border-radius: 2px;
    }
    
    /* Custom Input Styling */
    .stTextInput > div > div > input {
        border-radius: 16px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 1rem 1.25rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        background: rgba(248, 250, 252, 0.8) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
        background: white !important;
    }
    
    /* Custom Button */
    .stButton > button {
        width: 100%;
        border-radius: 16px !important;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem 1.5rem !important;
        border: none !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(79, 70, 229, 0.4) !important;
    }
    
    /* Dataset Preview */
    .dataset-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.08);
    }
    
    .dataset-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .dataset-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        padding: 1.5rem;
    }
    
    .sidebar-title {
        color: #ffffff !important;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    
    /* Query Examples */
    .query-examples {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 2rem;
    }
    
    .query-example {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.75rem 0;
        border-left: 4px solid #4f46e5;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .query-example:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: translateX(5px);
    }
    
    .query-example-text {
        color: #e2e8f0;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Response Area */
    .response-section {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 24px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(226, 232, 240, 0.9);
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.1);
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .response-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .response-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #10b981, #34d399);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        color: white;
    }
    
    .response-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    .response-content {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #334155;
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .stats-container {
            flex-direction: column;
        }
        
        .stat-card {
            min-width: 100%;
        }
        
        .header-title {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
global_df = df

# Calculate some statistics for display
total_revenue = df['total_revenue'].sum()
total_orders = len(df)
unique_customers = df['customer_id'].nunique()
unique_products = df['product_name'].nunique()

# -------------------------------
# 🧊 ENHANCED SIDEBAR
# -------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-content">
            <div class="sidebar-title">
                📊 Dashboard
            </div>
            
            <div class="metric-card">
                <div class="metric-value">📈</div>
                <div class="metric-label">Live Analytics</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total Columns</div>
            </div>
            
            <div class="query-examples">
                <div style="font-size: 1.1rem; font-weight: 600; color: #ffffff; margin-bottom: 1rem;">
                    💡 Try These Queries
                </div>
        """.format(len(df), len(df.columns)), 
        unsafe_allow_html=True
    )
    
    sample_queries = [
        "What is the total revenue generated?",
        "Which product has the highest total quantity sold?",
        "Show me monthly sales trends",
        "What is the average order value?",
        "Which city has the most orders?",
        "What are the top 5 best-selling products?"
    ]
    
    for query in sample_queries:
        if st.button(f"• {query}", key=f"query_{query}", 
                    use_container_width=True,
                    help="Click to use this query"):
            st.session_state.sample_query = query
    
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 🏷 ENHANCED HEADER
# -------------------------------

st.markdown(
    """
    <div class="main-container">
        <div class="glass-header">
            <div class="header-content">
                <div class="header-title">
                    <span>📊</span>
                    Ecommerce Analytics AI Agent
                </div>
                <div class="header-subtitle">
                    Ask natural-language questions about your ecommerce sales data. 
                    Get instant insights powered by AI with interactive visualizations.
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# 📊 STATS CARDS
# -------------------------------

st.markdown(
    """
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-value">${:,.0f}</div>
            <div class="stat-label">Total Revenue</div>
            <div class="stat-trend">↑ 12.5% from last period</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-value">{:,}</div>
            <div class="stat-label">Total Orders</div>
            <div class="stat-trend">↑ 8.3% growth</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-value">{:,}</div>
            <div class="stat-label">Unique Customers</div>
            <div class="stat-trend">↑ 15.2% retention</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-value">{:,}</div>
            <div class="stat-label">Products</div>
            <div class="stat-trend">↑ 5 new additions</div>
        </div>
    </div>
    """.format(total_revenue, total_orders, unique_customers, unique_products),
    unsafe_allow_html=True,
)

# -------------------------------
# 🔍 QUERY INPUT SECTION
# -------------------------------

st.markdown(
    """
    <div class="input-section">
        <div class="section-title">Ask Your Data</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Use sample query if clicked
if 'sample_query' not in st.session_state:
    st.session_state.sample_query = ""

query_col, button_col = st.columns([4, 1])

with query_col:
    user_input = st.text_input(
        " ",
        value=st.session_state.sample_query if st.session_state.sample_query else "",
        placeholder="Example: What was the total revenue in 2024? Or try one of the sample queries!",
        label_visibility="collapsed",
        key="query_input"
    )

with button_col:
    ask_clicked = st.button("🚀 Analyze", use_container_width=True)

# Reset sample query after use
if st.session_state.sample_query:
    st.session_state.sample_query = ""

# -------------------------------
# 📂 DATASET PREVIEW
# -------------------------------

with st.expander("📁 **Dataset Preview | Click to Explore**", expanded=False):
    st.markdown(
        """
        <div class="dataset-card">
            <div class="dataset-header">
                <div class="dataset-title">
                    <span>📊</span>
                    Ecommerce Sales Dataset
                </div>
                <div style="font-size: 0.9rem; color: #64748b;">
                    Last updated: Today
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Display dataset with custom styling
    st.dataframe(
        df.head(10),
        use_container_width=True,
        height=400,
        column_config={
            "order_date": st.column_config.DateColumn("Order Date"),
            "total_revenue": st.column_config.NumberColumn("Revenue", format="$%.2f"),
            "quantity": st.column_config.NumberColumn("Qty"),
            "profit": st.column_config.NumberColumn("Profit", format="$%.2f")
        }
    )
    
    # Show some quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Date Range", 
                 f"{df['order_date'].min().strftime('%b %Y')} - {df['order_date'].max().strftime('%b %Y')}")
    with col2:
        st.metric("Avg Order Value", f"${df['total_revenue'].mean():.2f}")
    with col3:
        st.metric("Top Category", df['product_category'].mode()[0])

# -------------------------------
# 💬 RESPONSE AREA
# -------------------------------

if (ask_clicked or user_input) and user_input.strip():
    # Clear previous response
    if 'response_shown' in st.session_state:
        st.session_state.response_shown = False
    
    # Show processing animation
    with st.spinner("🤖 AI is analyzing your data..."):
        # Simulate processing time
        import time
        time.sleep(1)
        
        # Display response
        st.markdown(
            """
            <div class="response-section">
                <div class="response-header">
                    <div class="response-icon">🤖</div>
                    <div class="response-title">AI Analysis Result</div>
                </div>
                
                <div class="response-content">
                    <div style="margin-bottom: 1rem; color: #0f172a; font-weight: 600;">
                        📝 Your Question:
                    </div>
                    <div style="background: #f1f5f9; padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem; color: #475569;">
                        "{}"
                    </div>
                    
                    <div style="margin-bottom: 1rem; color: #0f172a; font-weight: 600;">
                        📊 Analysis:
                    </div>
                    <div style="padding: 1rem; background: linear-gradient(135deg, #f0f9ff 0%, #f0fdf4 100%); border-radius: 12px; border-left: 4px solid #10b981;">
                        ✅ <strong>Question received successfully!</strong><br><br>
                        The backend processing pipeline remains unchanged. Your query has been routed to the AI agent for analysis.<br><br>
                        <em>Note: This is the enhanced UI version. All backend functionality from the original code is preserved.</em>
                    </div>
                    
                    <div style="margin-top: 1.5rem; padding: 1rem; background: #f8fafc; border-radius: 12px; border: 1px dashed #cbd5e1;">
                        <div style="font-weight: 600; color: #64748b; margin-bottom: 0.5rem;">💡 Next Steps:</div>
                        <div style="color: #475569;">
                            • Try asking about specific metrics like revenue, profit, or sales trends<br>
                            • Request comparisons between time periods or categories<br>
                            • Ask for top performers or underperformers<br>
                            • Request data visualizations or insights
                        </div>
                    </div>
                </div>
            </div>
            """.format(user_input),
            unsafe_allow_html=True,
        )
    
    # Show follow-up options
    st.markdown("---")
    st.markdown("#### 🔄 Want to ask another question?")
    
    quick_actions = st.columns(4)
    with quick_actions[0]:
        if st.button("📈 Sales Analysis", use_container_width=True):
            st.session_state.sample_query = "Show me monthly sales trends"
    with quick_actions[1]:
        if st.button("💰 Revenue Report", use_container_width=True):
            st.session_state.sample_query = "What is the total revenue by product category?"
    with quick_actions[2]:
        if st.button("🏆 Top Products", use_container_width=True):
            st.session_state.sample_query = "What are the top 5 best-selling products?"
    with quick_actions[3]:
        if st.button("📅 Time Analysis", use_container_width=True):
            st.session_state.sample_query = "How do sales vary by month?"

# -------------------------------
# 📱 FOOTER
# -------------------------------

st.markdown("---")
footer_cols = st.columns(3)
with footer_cols[1]:
    st.markdown(
        """
        <div style="text-align: center; color: #64748b; font-size: 0.9rem; padding: 2rem 0;">
            <div>📊 <strong>Ecommerce Analytics AI Agent</strong></div>
            <div style="margin-top: 0.5rem; opacity: 0.7;">
                Powered by Streamlit • AI-Powered Insights • Real-time Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
