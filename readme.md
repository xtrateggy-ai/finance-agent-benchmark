# Finance-Agent-Benchmark Enhancement

> **Berkeley Agentic AI Class Assignment**  
> Enhancing Vals.ai's Finance-Agent-Benchmark with AgentBeats integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

## Overview

This project enhances the original [Vals.ai Finance-Agent-Benchmark](https://www.vals.ai/benchmarks/finance_agent) by integrating it with Berkeley's [AgentBeats](https://github.com/agentbeats) framework. The enhancement—referred to as "agentification"—transforms the original single-agent benchmark into a multi-agent system using:

- **A2A Protocol**: Agent-to-agent communication
- **MCP (Model Context Protocol)**: Dynamic tool discovery and execution
- **Green/Purple Agent Architecture**: Evaluator and executor pattern

## Architecture

White agent is same as Purple agent

<img width="1406" height="806" alt="Screenshot From 2025-12-31 11-01-52" src="https://github.com/user-attachments/assets/be40d239-8f6b-417d-8c02-e551be7aa346" />

```
AgentBeats Platform
 │
 └── Calls Green Agent (A2A) → http://green:9000/a2a
      │
      ├── /reset  - Reset agent state
      ├── /health - Health check
      └── /a2a    - Assessment tasks
      
Launcher
 │
 ├──→ Green Agent (Port 9000)
 │     ├── A2A Server: http://green:9000/a2a
 │     ├── MCP Server: http://green:9001/sse
 │     ├── Orchestrates assessment
 │     └── Exposes tools via MCP
 │
 └──→ White Agent (Port 8000)
       ├── A2A Server: http://white:8000/a2a
       ├── MCP Client: Discovers tools from Green
       ├── LLM Reasoner: Decides which tools to call
       └── Executes tool calls via MCP
```

## Original Benchmark Resources

### Vals.ai Finance-Agent-Benchmark

- **Website**: [vals.ai/benchmarks/finance_agent](https://www.vals.ai/benchmarks/finance_agent)
  - Benchmark overview, design philosophy, and evaluation metrics
  - Task examples and tool descriptions
  
- **GitHub**: [vals-ai/finance-agent](https://github.com/vals-ai/finance-agent)
  - Source code for `run_agent.py` (main evaluation script)
  - Implementation of 4 core tools

### Core Tools (Enhanced for MCP)

| Tool | Description | Requirements |
|------|-------------|--------------|
| `company CIK resolver` | get company CIK based on the company name | `setup the user and email in env` |
| `xbrl company facts` | returns all the company concepts data for a company | `setup the user and email in env` |
| `xbrl company concept` | returns all the XBRL disclosures from a single company (CIK) and concept (a taxonomy and tag) into a single JSON file, with a separate array of facts for each units on measure that the company has chosen to disclose | `setup the user and email in env` |
| `xbrl frames` | The xbrl/frames API returns the most recent filed fact per entity for a requested time period | `setup the user and email in env` |
| `sec search rag` | Local Rag that fetch submissions(10-K, 10-Q, 8-K, DEF-14A), embed and save them to answer relative questions | `setup the required env var` |
| `yfinance` | Helper tool to calculate financial metrics | `setup the required env var` |
| `get today date` | Helper tool that helps LLMs understand latest data | `setup the required env var` |

some APIs resource: https://www.sec.gov/search-filings/edgar-application-programming-interfaces

## AgentBeats Integration

Based on the [AgentBeats Blog Series](https://docs.google.com/document/d/your-doc-id), this implementation follows the **Green/Purple Agent Pattern**:

### Green Agent (Evaluator)
- **Role**: Receives instructions from AgentBeats, orchestrates assessment
- **Endpoints**:
  - `GET /card` - Agent card (capabilities, skills)
  - `POST /a2a` - A2A message handling
  - `GET /health` - Health check
  - `POST /reset` - Reset agent state
- **MCP Server**: Exposes tools at `http://green:9001/sse`

### Purple Agent (Executor)
- **Role**: LLM-powered reasoner that discovers and uses tools
- **Process**:
  1. Receives question from Green Agent via A2A
  2. Discovers available tools via MCP (`tools/list`)
  3. Uses LLM to decide which tools to call
  4. Executes tool calls via MCP (`tools/call`)
  5. Returns answer to Green Agent
- **Features**:
  - Conversation memory for multi-turn reasoning
  - Dynamic tool discovery (no hardcoded tools)
  - Iterative refinement with fallback strategies

  This is implemented on the purple agent repository

### Agent Cards

Both agents expose `.well-known/agent-card.json` endpoints describing:
- Capabilities (streaming, multimodal support)
- Skills (financial analysis, web search, document parsing)
- Input/output modes (text, structured data)
- Version and metadata

## Dataset

**Public.csv or Financial-QA-10k**: Question-answer pairs about financial concepts and company data
- Location: `data/public.csv` (or `data/Financial-QA-10k.csv`) 
- Format: `question,answer` pairs
- Evaluation: Exact match accuracy

## Installation

### Prerequisites
- Python 3.13+
- Local LLM
- env file based on the env template
- setup required env variables:
   - `YourName`
   - `Email_ADDRESS`
   - `LLM_MODEL`
   - `LLM_API_KEY`
- API Keys:
  - `LLM_API_KEY` (Gemini/OpenAI/Anthropic)
    
### Other Setup
  ### LOCAL LLM & RAG
     USE_LOCAL_LLM_WHITE=0    # 1=True 0=False - White agent tool decisions (NEW)
     USE_LOCAL_LLM_JUDGE=0    # 1=True 0=False - Answer evaluation (NEW)
     USE_LOCAL_LLM_RAG=1      # 1=True 0=False - Choose between local LLM+RAG and Regex() extraction
     USE_LOCAL_LLM_GPU=1      # 1=True 0=False - Set use GPU for the local LLM, if available in the machine
     MAX_FILINGS_PER_QUESTION=125  # Max number of filings to process per question.
     LOCAL_LLM_MODEL_PATH=models/qwen2.5-3b-instruct-q4_k_m.gguf
     #LOCAL_LLM_MODEL_PATH=models/llama-3.2-3b-instruct-q4_k_m.gguf  # NOTE: this is 3B (not 1B) parameers
     LOCAL_LLM_JUDGE_MODEL=models/llama-3.2-1b-instruct-q4_k_m.gguf
     
     LOCAL_LLM_WHITE_CONTEXT=6144   # White agent (needs more for tool descriptions)
     LOCAL_LLM_JUDGE_CONTEXT=2048   # Judge (just comparing two strings)
     LOCAL_LLM_RAG_CONTEXT=4096     # RAG extraction
### Local Setup

```bash
# Clone repository
git clone https://github.com/your-username/finance-agent-benchmark.git
cd finance-agent-benchmark

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

# Configure secrets
cp secrets/secrets.env.example secrets/secrets.env
# Edit secrets/secrets.env with your API keys and setup
```

### Configuration

Edit `secrets/secrets.env`:

```bash
# LLM Configuration
LLM_MODEL=gemini/gemini-2.5-flash-lite
LLM_API_KEY=your_api_key_here

# Agent Configuration
GREEN_AGENT_HOST=127.0.0.1
GREEN_AGENT_PORT=9000
WHITE_AGENT_HOST=127.0.0.1
WHITE_AGENT_PORT=8000
MCP_PORT=9001

# Dataset
DATASET=data/public.csv
#DATASET=data/Financial-QA-10k.csv
NUM_TASKS=5

# Optional: Tool API Keys
SERP_API_KEY=your_serpapi_key
SEC_API_KEY=your_sec_api_key
```

## Usage

### Manually

```bash
# Terminal 1: Start Green Agent
cd app
python green_agent_mcp_a2a.py

# Terminal 2: Start White Agent
cd app
python white_agent_mcp_memory.py

# Terminal 3: Run Assessment
cd app
python launcher.py --num_tasks 5
```

### Using Launcher (Recommended)

```bash
# Start both agents and run assessment
cd app
python launcher.py --num_tasks 5

# Options:
#   --num_tasks N       Number of questions to evaluate
#   --green_host HOST   Green agent host (default: 127.0.0.1)
#   --green_port PORT   Green agent port (default: 9000)
#   --white_host HOST   White agent host (default: 127.0.0.1)
#   --white_port PORT   White agent port (default: 8000)
```

## Project Structure

```
finance-agent-benchmark
├── app
│   ├── cards
│   │   ├── green_card.toml
│   │   └── white_card.toml
│   ├── data
│   │   ├── public.csv  
│   ├── eval_result.csv
│   ├── eval_white.txt
│   ├── green_agent_mcp_a2a_judge_rag.py
│   ├── kill_agentbeats.sh
│   ├── launcher.py
│   ├── main.py   
│   ├── run.bat
│   ├── run_launcher.sh
│   ├── run.sh
│   ├── secrets
│   │   ├── secrets.env
│   │   └── secrets.env.example
│   ├── tools
│   │   ├── company_CIK.py
│   │   ├── company_tickers.json
│   │   ├── edgar_submissions.py
│   │   ├── __init__.py
│   │   ├── local_llm_rag.py
│   │   ├── models
│   │   │   └── Llama-3.2-1B-Instruct-Q4_K_M.gguf
│   │   ├── sec_search_rag.py
│   │   ├── today_date.py
│   │   ├── xbrl_company_concept.py
│   │   ├── xbrl_company_facts.py
│   │   ├── xbrl_frames.py
│   │   └── yfinance_search.py
│   ├── utils
│   │   ├── env_setup.py
│   │   ├── llm_judge_old.py
│   │   ├── llm_judge.py
│   │   ├── llm_manager.py
│   │   ├── local_llm_wrapper.py
│   └── white_agent_mcp_memory.py
├── readme.md
└── requirements.txt

```

## Key Features

### 1. Dynamic Tool Discovery
- White agent discovers tools via MCP `tools/list` at runtime
- No hardcoded tool knowledge required
- Supports adding new tools without code changes

### 2. Conversation Memory
- White agent maintains conversation history
- Tracks tool calls and results
- Enables multi-turn reasoning

### 3. Error Resilience
- Automatic fallback to alternative tools
- Handles API failures gracefully
- Retries with exponential backoff

### 4. AgentBeats Compliance
- A2A protocol for agent communication
- Agent cards expose capabilities
- Health checks and reset endpoints
- Stateless assessment runs

## Evaluation Metrics

Match the answers rubrics that is on the dataset semantically. These rubrics are different based on each question's category and a series of steps to answer a particular question.

## References

- **Vals.ai Benchmark**: [vals.ai/benchmarks/finance_agent](https://www.vals.ai/benchmarks/finance_agent)
- **Vals.ai GitHub**: [github.com/vals-ai/finance-agent](https://github.com/vals-ai/finance-agent)
- **AgentBeats**: [github.com/agentbeats](https://github.com/agentbeats)
- **MCP Specification**: [modelcontextprotocol.io](https://modelcontextprotocol.io)
- **A2A Protocol**: [Google A2A Announcement](https://developers.google.com/a2a)
- **Edgar APIs**: [SEC.gov APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- **agent Beates Docs**: [Berkeley RDI platform](https://docs.agentbeats.org/)

## Contributing

This is an academic project for UC Berkeley's Agentic AI class. Contributions should align with the assignment requirements.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Vals.ai** for the original Finance-Agent-Benchmark
- **UC Berkeley** for the AgentBeats framework
- **Course Instructors** for guidance and support

## Contact

- Fabio - inphlection@gmail.com
- Kiarash - kiarash996@gmail.com
- Milad - milad.eslamzadeh@teleperformance.com

---

**Status**: ✅ Development Complete | 🚀 Ready for AgentBeats Submission
