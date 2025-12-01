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
- **Green/White Agent Architecture**: Evaluator and executor pattern

## Project Goals

1. **Adapt** the original `run_agent.py` script and its 4 tools
2. **Integrate** with AgentBeats using A2A protocol for agent communication
3. **Expose** tools dynamically via MCP for the white agent to discover
4. **Evaluate** financial question-answering capabilities on a Financial Dataset

## Architecture

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
| `google_web_search` | Web search via SerpAPI | `SERP_API_KEY` |
| `edgar_search` | SEC EDGAR filings search | `SEC_API_KEY` |
| `parse_html_page` | Parse and chunk HTML documents | None |
| `retrieve_information` | Query extracted text with LLM | `LLM_API_KEY` |

## AgentBeats Integration

Based on the [AgentBeats Blog Series](https://docs.google.com/document/d/your-doc-id), this implementation follows the **Green/White Agent Pattern**:

### Green Agent (Evaluator)
- **Role**: Receives instructions from AgentBeats, orchestrates assessment
- **Endpoints**:
  - `GET /card` - Agent card (capabilities, skills)
  - `POST /a2a` - A2A message handling
  - `GET /health` - Health check
  - `POST /reset` - Reset agent state
- **MCP Server**: Exposes tools at `http://green:9001/sse`

### White Agent (Executor)
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
- Python 3.12+
- Docker (optional, for containerized deployment)
- API Keys:
  - `LLM_API_KEY` (Gemini/OpenAI/Anthropic)
  - `SERP_API_KEY` (optional, for web search)
  - `SEC_API_KEY` (optional, for EDGAR search)

### Local Setup

```bash
# Clone repository
git clone https://github.com/your-username/finance-agent-benchmark.git
cd finance-agent-benchmark

# Create virtual environment
python -m venv fab
source fab/bin/activate  # On Windows: fab\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure secrets
cp secrets/secrets.env.example secrets/secrets.env
# Edit secrets/secrets.env with your API keys
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

### Local Development

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

### Docker Deployment

```bash
# Build image
docker build -t finance-agent-benchmark .

# Run single container (both agents)
docker run --rm \
  -p 9000:9000 -p 9001:9001 -p 8000:8000 \
  -e LLM_API_KEY=your_key \
  -e NUM_TASKS=5 \
  finance-agent-benchmark

# Or use environment file
docker run --rm \
  -p 9000:9000 -p 9001:9001 -p 8000:8000 \
  --env-file secrets/secrets.env \
  finance-agent-benchmark

# View logs
docker logs -f finance-agent-benchmark
```

### Testing Endpoints

```bash
# Green Agent
curl http://localhost:9000/health
curl http://localhost:9000/card
curl http://localhost:9001/sse  # MCP endpoint

# White Agent
curl http://localhost:8000/health
curl http://localhost:8000/card

# Test question via A2A
curl -X POST http://localhost:8000/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is revenue?",
    "mcp_url": "http://localhost:9001"
  }'
```

## Project Structure

```
finance-agent-benchmark/
├── app/
│   ├── green_agent_mcp_a2a.py      # Green agent (orchestrator + MCP server)
│   ├── white_agent_mcp_memory.py   # White agent (reasoner + MCP client)
│   ├── launcher.py                 # Main launcher script
│   ├── cards/
│   │   ├── green_card.toml         # Green agent capabilities
│   │   └── white_card.toml         # White agent capabilities
│   ├── tools/
│   │   ├── google_search.py        # Web search tool
│   │   ├── serp_search.py          # SERP API search
│   │   ├── edgar_search.py         # SEC EDGAR search
│   │   ├── parse_html_page.py      # HTML parser
│   │   └── retrieve_information.py # Text retrieval tool
│   └── utils/
│       └── env_setup.py            # Environment configuration
├── data/
│   └── public.csv                  # Evaluation dataset (alternative: Financial-QA-10k.csv)
├── secrets/
│   ├── secrets.env.example         # Template for secrets
│   └── secrets.env                 # Your API keys (gitignored)
├── Dockerfile                      # Docker container definition
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
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

- **Primary Metric**: Match accuracy on answer in the dataset. 
- **Secondary Metrics**:
  - Tool call efficiency (calls per question)
  - Response time per question
  - Tool success rate

## Development

### Adding New Tools

1. Create tool implementation in `app/tools/`:
```python
# app/tools/my_tool.py
async def my_tool(param: str) -> dict:
    # Tool implementation
    return {"result": "data"}
```

2. Register in Green Agent MCP server:
```python
# In green_agent_mcp_a2a.py
@self.mcp.tool()
async def my_tool_handler(param: str) -> dict:
    """Tool description for LLM"""
    return await my_tool(param)
```

3. Tool is automatically discoverable by White Agent!

### Testing Tools

```bash
# Run tool test suite
cd app
python test_tools.py

# Tests each tool individually and verifies MCP integration
```

## Deployment

### Docker Hub

```bash
# Tag and push
docker tag finance-agent-benchmark your-dockerhub/finance-agent-benchmark
docker push your-dockerhub/finance-agent-benchmark
```

### Cloud Deployment (Nebius/AWS/GCP)

```bash
# Pull and run on cloud VM
docker pull your-dockerhub/finance-agent-benchmark
docker run -d \
  -p 9000:9000 -p 9001:9001 -p 8000:8000 \
  -e LLM_API_KEY=your_key \
  --restart unless-stopped \
  your-dockerhub/finance-agent-benchmark
```

### AgentBeats Platform

1. Deploy agents to cloud with public endpoints
2. Register on [agentbeats.ai](https://agentbeats.ai)
3. Submit agent URLs:
   - Green Agent: `https://your-domain.com/green`
   - MCP Endpoint: `https://your-domain.com/green/mcp`

## References

- **Vals.ai Benchmark**: [vals.ai/benchmarks/finance_agent](https://www.vals.ai/benchmarks/finance_agent)
- **Vals.ai GitHub**: [github.com/vals-ai/finance-agent](https://github.com/vals-ai/finance-agent)
- **AgentBeats**: [github.com/agentbeats](https://github.com/agentbeats)
- **MCP Specification**: [modelcontextprotocol.io](https://modelcontextprotocol.io)
- **A2A Protocol**: [Google A2A Announcement](https://developers.google.com/a2a)

## Contributing

This is an academic project for UC Berkeley's Agentic AI class. Contributions should align with the assignment requirements.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Vals.ai** for the original Finance-Agent-Benchmark
- **UC Berkeley** for the AgentBeats framework
- **Course Instructors** for guidance and support

## Contact

- **Course**: UC Berkeley Agentic AI
- **Assignment**: Finance Agent Benchmark Enhancement
- **Framework**: AgentBeats Integration

---

**Status**: ✅ Development Complete | 🚀 Ready for AgentBeats Submission