import streamlit as st
import asyncio
import json
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .moderate-risk {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the MCP client and LLM system"""
    try:
        llm = ChatOllama(model="gpt-oss:latest", base_url="http://172.21.0.189:11434")
        
        client = MultiServerMCPClient({
            "diabetes_server": {
                "url": "http://localhost:8080/mcp",
                "transport": "streamable_http"
            }
        })
        
        return llm, client
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None, None

async def check_mcp_connection(client):
    """Check if MCP server is connected"""
    try:
        tools = await client.get_tools()
        return True, len(tools)
    except Exception as e:
        return False, str(e)

async def get_prediction(llm, client, age, bmi, pedigree):
    """Get diabetes prediction using the MCP system"""
    try:
        print(f"🔄 Loading MCP tools...")
        mcp_tools = await client.get_tools()
        print(f"✅ Loaded {len(mcp_tools)} MCP tools")
        
        model = llm.bind_tools(mcp_tools)
        
        def should_continue(state):
            last_message = state["messages"][-1]
            return "tools" if last_message.tool_calls else END

        def call_model(state):
            print(f"📤 Sending request to LLM...")
            response = model.invoke(state["messages"])
            print(f"📥 Received response from LLM")
            return {"messages": [response]}

        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", ToolNode(mcp_tools))
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges("call_model", should_continue, ["tools", END])
        builder.add_edge("tools", "call_model")
        graph = builder.compile()

        query = f"Predict diabetes risk for age {age}, BMI {bmi}, pedigree {pedigree}"
        print(f"🚀 Starting prediction with query: {query}")
        
        result = await graph.ainvoke({
            "messages": [{"role": "user", "content": query}]
        })
        
        print(f"✅ Prediction completed successfully")
        return result["messages"][-1].content
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return f"Error getting prediction: {e}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Risk Predictor</h1>', unsafe_allow_html=True)
    
    # Initialize system
    llm, client = initialize_system()
    
    if llm is None or client is None:
        st.error("System initialization failed. Please check if the MCP server is running.")
        st.stop()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Patient Information")
        
        age = st.slider("Age (years)", 18, 100, 45, help="Patient's age in years")
        bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 25.0, 0.1, help="Body Mass Index")
        pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01, 
                           help="Genetic predisposition score based on family history")
        
        predict_button = st.button("🔍 Predict Risk", type="primary", use_container_width=True)
        
        # MCP Connection Status
        st.markdown("---")
        st.subheader("🔗 System Status")
        
        with st.spinner("Checking MCP connection..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                is_connected, info = loop.run_until_complete(check_mcp_connection(client))
                loop.close()
                
                if is_connected:
                    st.success(f"✅ MCP Server Connected ({info} tools available)")
                else:
                    st.error(f"❌ MCP Server Disconnected: {info}")
            except Exception as e:
                st.error(f"❌ Connection Check Failed: {e}")
        
        # Risk factor guidelines
        with st.expander("📋 Risk Factor Guidelines"):
            st.markdown("""
            **Age Risk Levels:**
            - Low: < 45 years
            - Moderate: 45-54 years  
            - High: > 55 years
            
            **BMI Categories:**
            - Normal: 18.5-24.9
            - Overweight: 25-29.9
            - Obese: > 30
            
            **Pedigree Function:**
            - Low: < 0.3
            - Moderate: 0.3-0.6
            - High: > 0.6
            """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            with st.spinner("Analyzing patient data..."):
                try:
                    print(f"🎯 Starting prediction for Age: {age}, BMI: {bmi}, Pedigree: {pedigree}")
                    
                    # Run async prediction
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(get_prediction(llm, client, age, bmi, pedigree))
                    loop.close()
                    
                    print(f"📊 Raw prediction result: {result[:100]}...")
                    
                    # Display result
                    st.subheader("📊 Prediction Results")
                    
                    # Parse result to extract key information
                    if "high risk" in result.lower() or "diabetes" in result.lower():
                        risk_class = "high-risk"
                        risk_level = "High Risk"
                        risk_color = "🔴"
                    elif "moderate" in result.lower():
                        risk_class = "moderate-risk"
                        risk_level = "Moderate Risk"
                        risk_color = "🟡"
                    else:
                        risk_class = "low-risk"
                        risk_level = "Low Risk"
                        risk_color = "🟢"
                    
                    print(f"🏷️ Classified as: {risk_level}")
                    
                    st.markdown(f"""
                    <div class="prediction-box {risk_class}">
                        <h3>{risk_color} {risk_level}</h3>
                        <p><strong>AI Analysis:</strong></p>
                        <p>{result}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    print(f"❌ Prediction failed with error: {e}")
                    st.error(f"Prediction failed: {e}")
        else:
            st.info("👈 Enter patient information in the sidebar and click 'Predict Risk' to get started.")
    
    with col2:
        st.subheader("📈 Current Input Summary")
        
        # Display current inputs in a nice format
        st.metric("Age", f"{age} years")
        st.metric("BMI", f"{bmi:.1f} kg/m²")
        st.metric("Pedigree Function", f"{pedigree:.2f}")
        
        # Risk assessment based on individual factors
        st.subheader("🎯 Individual Risk Factors")
        
        # Age assessment
        if age < 45:
            st.success("Age: Low Risk")
        elif age <= 54:
            st.warning("Age: Moderate Risk")
        else:
            st.error("Age: High Risk")
        
        # BMI assessment
        if bmi < 25:
            st.success("BMI: Normal")
        elif bmi < 30:
            st.warning("BMI: Overweight")
        else:
            st.error("BMI: Obese")
        
        # Pedigree assessment
        if pedigree < 0.3:
            st.success("Genetics: Low Risk")
        elif pedigree <= 0.6:
            st.warning("Genetics: Moderate Risk")
        else:
            st.error("Genetics: High Risk")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Disclaimer:</strong> This tool is for educational purposes only. 
        Always consult healthcare professionals for medical decisions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
