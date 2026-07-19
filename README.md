# PSI20 AI Market Analysis

An AI-assisted financial analysis and short-term stock forecasting system developed during my internship at BI4ALL as part of the final semester of my Bachelor's degree.

The project combines quantitative time-series forecasting with qualitative financial-news analysis using a multi-agent AI architecture. The system is designed to analyze companies belonging to the PSI20 index and generate a final market outlook ranging from **Bearish** to **Bullish**, together with a confidence level and risk considerations.

> **Disclaimer:** This project is an academic and research-oriented system. Its outputs should not be interpreted as financial advice or as a guarantee of future market performance.

---

## Overview

Financial markets are influenced by both historical price patterns and constantly changing external events. This project explores a hybrid approach that combines these two perspectives.

The system consists of two main analytical components:

### Quantitative Analysis

The quantitative pipeline uses an LSTM-based deep learning model to analyze historical stock-price data and generate short-term forecasts.

The forecasting component also estimates predictive uncertainty using Monte Carlo Dropout, allowing the system to consider not only the expected price trajectory but also the uncertainty surrounding the prediction.

The quantitative analysis considers factors such as:

* Historical stock-price behaviour
* Time-series patterns
* Technical indicators and engineered features
* Short-term forecast trajectories
* Predictive uncertainty
* Risk measures such as Value at Risk (VaR) and Conditional Value at Risk (CVaR)

### Qualitative Analysis

The qualitative pipeline uses a multi-agent AI architecture to analyze recent financial news and other market information.

The agents are specialized into different analytical roles, including:

* **News Analysis** — Extracts structured events and relevant information from financial news.
* **News Deduplication** — Identifies multiple articles describing the same underlying event.
* **Market Sentiment Analysis** — Evaluates the potential market impact and relevance of news events.
* **Quantitative Forecast Analysis** — Interprets the outputs of the forecasting model and its uncertainty estimates.
* **Investment Strategy and Decision-Making** — Combines the quantitative and qualitative signals into a final structured market outlook.

The final analysis produces a recommendation ranging from:

**Bearish → Cautelously Bearish → Neutral → Cautelously Bullish → Bullish**

along with an associated confidence level and risk considerations.

---

## System Architecture

The general pipeline can be summarized as follows:

```text
Historical Market Data
        │
        ▼
Data Processing & Feature Engineering
        │
        ▼
LSTM Time-Series Forecasting
        │
        ├──► Price Forecast
        ├──► Uncertainty Estimation
        └──► Risk Metrics
        │
        ▼
Recent Financial News
        │
        ▼
Multi-Agent AI Analysis
        │
        ├──► Event Extraction
        ├──► News Deduplication
        ├──► Sentiment & Impact Analysis
        └──► Quantitative Forecast Interpretation
        │
        ▼
Final Decision Agent
        │
        ▼
Bullish-to-Bearish Market Outlook
```

The project also includes a Streamlit interface that integrates the forecasting and multi-agent analysis components into a single interactive application.

---

## Technologies

The project was developed using:

* **Python 3.12.1**
* **TensorFlow / Keras** — LSTM-based forecasting
* **CrewAI** — Multi-agent orchestration
* **Large Language Models** — Financial-news analysis and decision synthesis
* **Streamlit** — Interactive web interface
* **Plotly** — Data visualization
* **Yahoo Finance** — Historical market data
* **DuckDuckGo Search** — Automated financial-news retrieval
* **Jupyter Notebooks** — Research, experimentation, and development

The exact dependencies are listed in:

```text
requirements.txt
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Jorge-Couto/psi20-ai-market-analysis.git
cd psi20-ai-market-analysis
```

## 2. Create a Virtual Environment

It is recommended to use a virtual environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install the Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

The multi-agent component requires access to a compatible Large Language Model.

The repository does **not** include the `.env` file because it may contain private API keys and credentials.

Create a file named:

```text
.env
```

in the root directory of the project.

The exact variables required depend on the LLM provider being used.

---

## Option A — OpenAI-Compatible API

For an OpenAI or OpenAI-compatible API:

```env
API_KEY=your_api_key_here
BASE_URL=https://api.openai.com/v1
```

Depending on the provider and implementation, the model configuration may look like:

```python
llm = LLM(
    model="your-model-name",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
```

---

## Option B — Azure OpenAI

For an Azure-hosted model:

```env
API_KEY=your_azure_api_key_here
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
```

The LLM configuration may look like:

```python
llm = LLM(
    model="your-deployment-name",
    api_key=os.getenv("API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version="2024-12-01-preview",
)
```

---

## Option C — Other Compatible Providers

The system can be adapted to other providers depending on the API format supported by the LLM framework.

For example:

```python
llm = LLM(
    model="your-model-name",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)
```

The exact environment variables and configuration may need to be adapted according to the provider being used.

---

# Data Setup

The project is designed to obtain the required market data automatically.

If running the full pipeline from the beginning using the Jupyter Notebooks, first run:

```text
Data_Downloads.ipynb
```

This notebook downloads and prepares the datasets required by the subsequent forecasting and analysis stages.

The project intentionally does not include generated datasets or unnecessary local data files in the repository.

---

# Running the Project

There are three main ways to run the project.

## Option 1 — Jupyter Notebooks

The notebooks represent the research and development workflow.

Run them in the following order:

### 1. Download and Prepare the Data

```text
Data_Downloads.ipynb
```

### 2. Run the Quantitative Forecasting Pipeline

```text
Time_Series_Forecasting_Final.ipynb
```

This notebook performs the time-series forecasting process.

### 3. Run the AI-Agent Analysis

```text
AI_Agents.ipynb
```

This notebook executes the multi-agent analysis and generates the qualitative and final decision outputs.

---

## Option 2 — Python Script

The main pipeline can be executed through:

```bash
python main.py
```

This version is intended to provide a more structured script-based execution of the project pipeline.

The modular components used by the script are located in:

```text
my_app/
├── Agents_Runner.py
├── Forecasting.py
└── Streamlit.py
```

---

## Option 3 — Streamlit Application

To launch the interactive web application:

```bash
streamlit run my_app/Streamlit.py
```

The application allows the user to:

1. Select a company from the PSI20 index.
2. Retrieve and visualize historical stock-price data.
3. Generate a short-term quantitative forecast.
4. Inspect forecast uncertainty.
5. Execute the multi-agent financial-news analysis.
6. Obtain a final structured market outlook.

The Streamlit application acts as an interactive interface over the underlying forecasting and multi-agent components.

> **Important:** Run the Streamlit command from the root directory of the repository. Do not execute the scripts directly from inside the `my_app` directory.

---

# Project Structure

```text
psi20-ai-market-analysis/
│
├── AI_Agents.ipynb
│   └── Development and execution of the multi-agent analysis pipeline.
│
├── Data_Downloads.ipynb
│   └── Downloads and prepares the required market data.
│
├── Time_Series_Forecasting_Final.ipynb
│   └── Development and execution of the quantitative forecasting pipeline.
│
├── main.py
│   └── Script-based execution of the main project pipeline.
│
├── my_app/
│   ├── Agents_Runner.py
│   │   └── Modularized execution of the AI-agent pipeline.
│   │
│   ├── Forecasting.py
│   │   └── Modularized quantitative forecasting functionality.
│   │
│   └── Streamlit.py
│       └── Interactive Streamlit application.
│
├── requirements.txt
│   └── Python dependencies.
│
├── .gitignore
│   └── Files and folders excluded from version control.
│
└── README.md
    └── Project documentation.
```

---

# Development Notes

The Jupyter Notebooks contain the implementation and execution workflow for the final forecasting and multi-agent analysis pipelines included in this repository.

They also include selected validation and experimentation steps, such as the validation of the parameter used to determine regime accuracy, where the regime threshold is defined as a multiple of volatility.

However, the notebooks do not represent the complete history of the project's research and experimentation. Several approaches explored during the internship, including traditional ARIMA and ARIMA-GARCH models, are not included. The notebooks are primarily focused on the selected final pipeline and its execution rather than documenting every alternative model, exploratory analysis, or development experiment conducted during the project.

For application-oriented execution, the core forecasting and agent functionality was refactored into Python modules:

* `Forecasting.py`
* `Agents_Runner.py`

This separation allows the Streamlit interface and the main script to use the core functionality without depending on the notebook-based workflow.

---

# Limitations

Financial markets are highly stochastic and influenced by many external variables that cannot be fully modelled.

The system should therefore be understood as an experimental decision-support system rather than a guaranteed prediction engine.

Important limitations include:

* Short-term stock-price forecasting is inherently uncertain.
* Historical patterns may not persist in future market conditions.
* News analysis depends on the availability and quality of retrieved information.
* LLM outputs may contain errors or incorrect interpretations.
* The empirical evaluation period was limited.
* Market conditions can change abruptly due to unexpected macroeconomic, geopolitical, or company-specific events.

---

# Future Work

Potential future improvements include:

* Walk-forward validation for the forecasting models.
* Improved calibration of predictive uncertainty.
* Additional macroeconomic and cross-asset features.
* Testing alternative forecasting architectures, including Gradient Boosting, ensemble methods, and temporal Transformers.
* Retrieval-Augmented Generation using annual reports and official company communications.
* Improved handling of conflicting or incomplete news information.
* Continuous feedback and online model retraining based on observed market reactions.

---

# Disclaimer

This project is intended for educational, research, and demonstration purposes only.

The forecasts, classifications, market outlooks, and other outputs generated by the system do not constitute financial advice, investment advice, or a recommendation to buy or sell any financial instrument.

Past market behaviour does not guarantee future results.
