# 🩺 Explainable Diabetes ML MCP: AI-Powered Healthcare Risk Assessment


## 🎯 Project Description

A cutting-edge **Explainable AI (XAI) healthcare platform** that revolutionizes diabetes risk prediction through the seamless integration of **Machine Learning**, **Model Context Protocol (MCP)**, and **interpretable AI technologies**. This system empowers healthcare professionals with transparent, evidence-based diabetes risk assessments backed by advanced ML models and real-time explainability features.

Built on a modern microservices architecture, the platform leverages **LangGraph AI agents**, **FastMCP protocol servers**, and **SHAP explainability frameworks** to deliver clinically-relevant insights through an intuitive web interface. The system transforms complex algorithmic predictions into actionable medical intelligence, ensuring healthcare decisions are both data-driven and interpretable.


### 🏗️ Technical Architecture

The system follows a **distributed, protocol-driven architecture** powered by **LangGraph agents**, **Model Context Protocol (MCP)**, and **explainable ML** for transparent healthcare AI workflows.

## LangGraph Agent Pipeline
- Orchestrates **medical prediction workflows** using `MessagesState` with conditional routing
- Executes **4 core nodes** — covering model invocation, tool execution, decision branching, and response formatting
- Employs **MCP tool binding** for seamless integration with diabetes prediction and explanation services
- Deterministic flow ensures clinical reliability (`START → call_model → tool_decision → execute_tools → END`)

## MCP Protocol Integration
- **Server:** FastMCP with HTTP/WebSocket transport on localhost:8080
- **Tools:** diabetes_risk_predictor and diabetes_risk_explainer with structured JSON schemas
- **Resources:** Medical guidelines and model metadata accessible via URI-based resource system
- **Communication:** Async HTTP with JSON-RPC for real-time tool discovery and execution

## LLM Integration
- **Model:** ChatOllama with gpt-oss:latest (configurable open-source medical LLM)
- **Inference Engine:** Ollama server with GPU acceleration on port 11434
- **Tool Binding:** Dynamic tool registration with automatic schema validation
- **Response Processing:** Structured medical analysis with risk classification and explanation parsing

## ML Processing Layer
- **Prediction Engine:** Scikit-learn ensemble models (Random Forest/XGBoost) with 85%+ accuracy
- **Explainability:** SHAP TreeExplainer for feature attribution and medical decision transparency
- **Data Pipeline:** NumPy arrays with validated medical parameter inputs (age, BMI, pedigree function)
- **Model Persistence:** Joblib serialization for consistent model versioning and deployment

## Healthcare Interface Layer
- **Frontend:** Streamlit with medical-grade UI components and real-time status monitoring
- **Visualization:** Risk classification with color-coded indicators and SHAP explanation rendering
- **Input Validation:** Medical parameter bounds checking with clinical guideline integration
- **Response Display:** Professional medical dashboard with disclaimer and compliance messaging

## Observability & Clinical Safety
- Comprehensive logging across all prediction steps with medical audit trails
- Real-time MCP connection monitoring for system reliability
- Input sanitization and error handling for clinical data safety
- Easily extensible to support additional medical models and healthcare protocols

### ⚙️ **Workflow Graph:**  
 
<img width="1884" height="1079" alt="Image" src="https://github.com/user-attachments/assets/900ecf39-1d97-419c-9c4a-61fd7686733d" />

---

### ✨ Key Features

- **🩺 Clinical Decision Support Interface**: Healthcare professionals can input patient parameters (age, BMI, family history) and receive instant diabetes risk assessments with confidence scores — no ML expertise required. The system translates medical inputs into actionable risk predictions automatically.

- **🔍 Explainable AI & Medical Reasoning**: Every prediction comes with SHAP-powered explanations showing exactly how each factor (age, BMI, genetics) contributes to the diabetes risk score. The system provides clear medical interpretations like "High BMI contributes 45% to elevated risk" using integrated explainability frameworks.

- **📊 Multi-Modal Risk Visualization**: Seamlessly presents both numerical predictions and visual risk indicators. The system dynamically renders color-coded risk classifications, individual factor assessments, and comprehensive medical dashboards based on the clinical context.

- **🔧 MCP Protocol Integration & Tool Transparency**: Every prediction utilizes standardized Model Context Protocol tools for diabetes assessment and explanation. Healthcare providers can see exactly which ML models and explainability methods were used, ensuring clinical transparency and audit compliance.

- **🔒 Healthcare-Grade Privacy & Compliance**: Runs entirely on local infrastructure using FastMCP and LangGraph, ensuring complete patient data privacy. Supports multiple medical AI models with HIPAA-compliant deployment options, configurable for different healthcare environments and regulatory requirements.

## 📁 Project Structure
```
📦 explainable-diabetes-ml-mcp
│
├── 📂 data/
│   └── pima_diabetes.csv           # Training dataset for diabetes prediction
│
├── 📂 models/
│   └── model.pkl                   # Pre-trained ML model (Random Forest)
│
├── mcp_server.py                   # FastMCP server with diabetes prediction tools
├── mcp_functions.py                # Core ML prediction and SHAP explanation functions
├── client.py                       # LangGraph agent client for testing MCP integration
├── streamlit_client_ui.py          # Professional Streamlit web application
├── train_model.ipynb               # Jupyter notebook for model training and evaluation
├── requirements.txt                # Python dependencies and package versions
└── README.md                       # Project documentation and setup guide


```



## 🧭 Demo Sample Images

### **Streamlit Interface**

<img width="1911" height="1010" alt="Image" src="https://github.com/user-attachments/assets/9675855f-ab0e-48c4-93e5-66f39b836392" />



### **FastMCP Server**

<img width="1915" height="513" alt="Image" src="https://github.com/user-attachments/assets/e6e8b734-25e4-4f20-8fbf-233dffeb97f3" />


---

## 🛠️ Installation Instructions

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (optional, for faster processing)
- 8GB+ RAM recommended

### Step 1: Clone Repository
```bash
git clone https://github.com/Ginga1402/explainable-diabetes-ml-mcp.git
cd explainable-diabetes-ml-mcp
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```


### Step 3: Set Up Ollama (LLM Backend)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the required model
ollama pull gpt-oss:latest
```



## 📖 Usage

### Starting the Application

1. **Start the FastMCP Server**:
```bash
python mcp_server.py
```
The API will be available at `http://localhost:8080`

2. **Launch the Streamlit Interface**:
```bash
streamlit run streamlit_client_ui.py
```
The web interface will open at `http://localhost:8501`



### 🧱 Technologies Used


| Technology | Description | Link |
|------------|-------------|------|
| **LangChain** | Framework for building LLM-driven applications and agent workflows | [LangChain](https://python.langchain.com) |
| **LangGraph** | State-based agent orchestration for complex medical AI workflows | [LangGraph](https://github.com/langchain-ai/langgraph) |
| **FastMCP** | Model Context Protocol server for distributed tool orchestration | [FastMCP](https://pypi.org/project/fastmcp/) |
| **ChatOllama** | Local LLM integration for privacy-focused healthcare AI | [Ollama](https://ollama.ai) |
| **gpt-oss:latest** | Open-source medical reasoning model for diabetes risk assessment | [GPT-OSS](https://ollama.com/library/gpt-oss) |
| **Scikit-learn** | Machine learning library for diabetes prediction models | [Scikit-learn](https://scikit-learn.org) |
| **SHAP** | Explainable AI library for medical decision transparency | [SHAP](https://shap.readthedocs.io) |
| **Streamlit** | Web framework for building interactive healthcare applications | [Streamlit](https://streamlit.io) |
| **Pandas** | Data manipulation and analysis for medical datasets | [Pandas](https://pandas.pydata.org) |
| **NumPy** | Fundamental package for scientific computing and ML operations | [NumPy](https://numpy.org) |
| **Joblib** | Model serialization and persistence for ML deployment | [Joblib](https://joblib.readthedocs.io) |
| **Asyncio** | Asynchronous programming for real-time medical AI responses | [Asyncio](https://docs.python.org/3/library/asyncio.html) |
| **Pydantic** | Data validation for medical parameter input safety | [Pydantic](https://pydantic.dev) |

## 🤝 Contributing

Contributions to this project are welcome! If you have ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Star History

If you find explainable-diabetes-ml-mcp useful, please consider giving it a star ⭐ on GitHub!
