# Finance-Agent-Benchmark Enhancement

> **UC Berkeley Agentic AI / AgentBeats Project**  
> Extending Vals.ai's Finance-Agent-Benchmark into an A2A-based, MCP-enabled financial agent evaluation system.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

## Overview

This project adapts the original [Vals.ai Finance-Agent-Benchmark](https://www.vals.ai/benchmarks/finance_agent) for the [AgentBeats](https://github.com/agentbeats) evaluation ecosystem. It separates **benchmark orchestration and scoring (Green Agent)** from **financial question answering (Purple Agent)**, allowing different participants to be assessed against a shared dataset and a common set of financial tools.

- **A2A (Agent-to-Agent) protocol:** Transports assessment requests and answers between independent agents.
- **MCP (Model Context Protocol):** Exposes SEC and financial-analysis tools for discovery and invocation by the participant.
- **Green/Purple architecture:** Keeps evaluation logic independent of the participant's implementation.
- **AgentBeats results:** Produces assessment metrics that can be published through the AgentBeats leaderboard workflow.

## Architecture

The **Green Agent** owns the benchmark questions, tool server, orchestration, and evaluation. The **Purple Agent** (previously called the *White Agent* in older versions) owns reasoning, tool selection, and answer generation. The Purple Agent may be implemented separately and does not need to share the Green Agent's codebase.

```text
AgentBeats / assessment client
        |
        | A2A: run_assessment(white_address, num_tasks)
        v
Green Agent — A2A server (port 9009)
  |  /.well-known/agent-card.json   /health   /a2a
  |
  +-- server.py       A2A request handling and service wiring
  +-- agent.py        Assessment entry point / coordination
  +-- executer.py     Dataset iteration, task execution, result aggregation
  +-- messenger.py    A2A requests to the participating Purple Agent
  +-- llm_judge.py    Answer evaluation and scoring
  |
  +-- MCP server (port 9001; transport endpoint depends on configuration)
          |
          +-- SEC EDGAR / XBRL APIs
          +-- SEC filing search and retrieval (RAG)
          +-- Financial metrics, ratios, and helper tools

                  A2A question + MCP endpoint
Green Agent ------------------------------------------> Purple Agent
Green Agent <------------------------------------------ Purple Agent
                       A2A answer                         |
                                                          | MCP tools/list,
                                                          | tools/call
                                                          v
                                                Green Agent MCP server

Green Agent -> judge -> structured assessment results -> AgentBeats leaderboard
```

**Evaluation flow:** The assessment client calls the Green Agent. Green loads a benchmark question and sends it to the configured Purple Agent over A2A, along with the MCP endpoint. Purple discovers and invokes tools as needed, then returns an answer over A2A. Green compares the answer with the reference answer using its judge, aggregates the results, and returns structured assessment output. The AgentBeats integration consumes those results for leaderboard reporting.

**Protocol responsibilities:** A2A is for *agent communication*; MCP is for *tool access*. This separation makes it possible to benchmark different Purple Agent implementations without changing the Green Agent's assessment interface.

## Original Benchmark Resources

### Vals.ai Finance-Agent-Benchmark

- **Website:** [vals.ai/benchmarks/finance_agent](https://www.vals.ai/benchmarks/finance_agent) — original benchmark context and financial tasks.
- **GitHub:** [vals-ai/finance-agent](https://github.com/vals-ai/finance-agent) — original implementation and tooling reference.

### Core Tools (Enhanced for MCP)

The Green Agent exposes financial tools through MCP. The precise registered tool names and arguments are defined by the running MCP server and can be discovered by a Purple Agent.

| Tool / capability | Purpose | Requirements |
|---|---|---|
| `cik_resolver` | Resolve company identifiers (CIKs) | SEC-compatible user-agent configuration |
| `companyfacts` | Retrieve company-wide XBRL facts | SEC access |
| `companyconcept` | Retrieve facts for a particular taxonomy concept | SEC access |
| `frames` | Retrieve XBRL facts for a reporting period | SEC access |
| `sec_search_rag` | Search SEC filings such as 10-K, 10-Q, 8-K, and DEF 14A | Filing access and configured retrieval backend |
| `get_financial_metrics` | Retrieve or compute financial metrics | Underlying financial data |
| `get_financial_ratios` | Retrieve or compute financial ratios | Underlying financial data |
| `get_ticker_symbol` / `get_company_name_from_ticker` | Resolve ticker and company names | Company mapping/data source |
| `today_date` | Supply the current date for time-sensitive tasks | No external API key expected |

SEC API reference: [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces). SEC access requires an appropriate identifying user-agent and respectful request rates.

## AgentBeats Integration

The implementation follows the **Green/Purple Agent pattern**: Green is the evaluator and tool provider; Purple is the participant being evaluated.

### Green Agent (Evaluator)

- **Role:** Accept assessment requests, load questions, communicate with the Purple Agent, expose MCP tools, judge answers, and aggregate scores.
- **A2A endpoints:**
  - `GET /.well-known/agent-card.json` — agent capabilities and discovery metadata.
  - `POST /a2a` — A2A JSON-RPC requests.
  - `GET /health` — health check.
- **A2A assessment operation:** `run_assessment`, with the participant address (`white_address`, retained as a compatibility argument name) and `num_tasks`.
- **MCP server:** Separate service on port `9001`; configure the actual transport URL (for example, an SSE endpoint when using SSE transport) for the deployment.
- **Implementation:** `server.py`, `agent.py`, `executer.py`, `messenger.py`, MCP registration, and the LLM judge.

### Purple Agent (Executor)

- **Role:** Answer the financial questions sent by Green. The reference Purple Agent is maintained separately and can be replaced with another compatible participant.
- **Process:**
  1. Receive the question and MCP endpoint through A2A.
  2. Connect to Green's MCP server and discover available tools.
  3. Select tools and investigate relevant financial data.
  4. Generate an answer using its own reasoning strategy and model.
  5. Return the answer to Green over A2A.
- **Implementation flexibility:** Purple Agents may use different LLMs, prompting strategies, retrieval workflows, or internal memory mechanisms. These are participant choices, not requirements imposed by Green.

Reference Purple image: `ghcr.io/kiarashyavari/finance-purple-agent:latest`.

### Agent Cards

The agents advertise their A2A identity and supported capabilities through `/.well-known/agent-card.json`. Agent cards describe the service URL, skills, input/output modes, and relevant metadata. Actual capabilities should be read from each running agent's card rather than assumed.

## Dataset

- **Default dataset:** `data/public.csv`.
- **Content:** Financial question/answer records used for assessment.
- **Alternative datasets:** Can be selected through `DATASET` when compatible with the evaluator's loader.
- **Evaluation:** Answers are judged against reference answers using an LLM-based evaluator designed to handle semantic and numerical equivalence; task-level outcomes are aggregated into accuracy metrics.

The benchmark dataset and judging logic belong to Green; a Purple Agent receives the question, not the reference answer.

## Installation

### Prerequisites

- Python 3.13+ for local development (match the repository's dependency constraints).
- Docker Engine and Docker Compose for containerized deployment.
- An LLM provider API key for the configured evaluation model.
- SEC user-agent identification for EDGAR requests.
- A reachable A2A-compatible Purple Agent.

### Other Setup

#### LLM, RAG, and cache settings

```dotenv
LLM_MODEL=gemini/gemini-2.5-flash-lite
LLM_API_KEY=your_llm_api_key
USE_LOCAL_LLM_RAG=1
USE_LOCAL_LLM_GPU=0
USE_DISK_CACHE=1
MAX_FILINGS_PER_QUESTION=50
```

The `USE_LOCAL_LLM_RAG` and GPU settings apply to the configured retrieval implementation, not to the Purple Agent's choice of LLM. Local model files are **not** a mandatory prerequisite for the Green Agent's LLM judge. Configure the SEC identity variables according to the current environment template.

### Local Setup

```bash
git clone https://github.com/KiarashYavari/finance-agent-benchmark.git
cd finance-agent-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create the environment file expected by your current launcher or Compose configuration, using the repository's example environment file where provided. Do not commit API keys or private `.env` files.

### Configuration

Representative Green Agent settings (confirm exact names in your current environment template):

```dotenv
HOST=0.0.0.0
AGENT_PORT=9009
MCP_PORT=9001
DATASET=data/public.csv
LLM_MODEL=gemini/gemini-2.5-flash-lite
LLM_API_KEY=your_llm_api_key
USE_DISK_CACHE=1
USE_LOCAL_LLM_RAG=1
USE_LOCAL_LLM_GPU=0
MAX_FILINGS_PER_QUESTION=50
```

The Purple Agent must be configured with the Green Agent's **reachable** MCP URL. Inside Docker Compose, service names may be used; from outside Docker, use a host-accessible address. `localhost` inside one container does not refer to another container.

## Usage

### Manually

Start the Green service with the repository's current entry point:

```bash
python launcher.py
```

Start an A2A-compatible Purple Agent separately, then invoke the Green Agent's `run_assessment` operation with its A2A address and the desired task count. The precise client command depends on the installed A2A/AgentBeats client version; the old `green_agent_mcp_a2a.py` and `white_agent_mcp_memory.py` commands are no longer the recommended entry points.

### Using Launcher (Recommended)

For the published Green Agent container:

```text
ghcr.io/kiarashyavari/finance-agent-benchmark:latest
```

For the reference Purple Agent container:

```text
ghcr.io/kiarashyavari/finance-purple-agent:latest
```

Use the repository's AgentBeats scenario/Compose configuration to start the services, provide environment variables, and execute the assessment. The Green A2A service uses port `9009` and its MCP service uses port `9001` in the current architecture. Confirm the exact Compose command and image tags in the checked-in deployment files before running.

Benchmark page: [AgentBeats Finance Agent Benchmark](https://agentbeats.dev/KiarashYavari/finance-agent-benchmark).

## Project Structure

The following is a **logical module map** of the newer architecture, not a claim that every file exists at exactly this path in every checkout:

```text
finance-agent-benchmark/
├── launcher.py                 # Green service entry point
├── src/
│   ├── server.py               # A2A server, routes, and MCP service wiring
│   ├── agent.py                # Assessment agent coordination
│   ├── executer.py             # Task execution and result aggregation
│   ├── messenger.py            # A2A communication with Purple
│   ├── mcp_tools.py            # MCP tool registration
│   ├── utils/
│   │   ├── llm_judge.py        # Semantic/numeric answer evaluation
│   │   └── llm_manager.py      # LLM configuration/helper logic
│   └── ...                     # SEC/XBRL/RAG tool implementations
├── data/
│   └── public.csv              # Benchmark questions and reference answers
├── cache/
│   └── sec/                    # Optional SEC filing cache
├── requirements.txt
└── README.md

# Purple Agent: independent repository and container
# ghcr.io/kiarashyavari/finance-purple-agent:latest
```

## Key Features

### 1. Dynamic Tool Discovery

- Green exposes a reusable suite of SEC and financial-analysis tools over MCP.
- Compatible Purple Agents can discover tool schemas at runtime.
- Tool implementations can evolve without embedding them in the participant's code.

### 2. Independent Agent Reasoning

- Purple controls its own reasoning process, tool selection, and answer generation.
- Different agent implementations can be evaluated against the same Green benchmark.
- Memory and multi-step reasoning are optional participant capabilities rather than Green Agent assumptions.

### 3. Error Resilience

- SEC requests can use retry/backoff for transient throttling and service errors.
- Optional disk caching reduces repeated SEC retrieval work.
- Separating the A2A and MCP interfaces makes service failures easier to isolate.

### 4. AgentBeats Compliance

- A2A-compatible agent card and assessment endpoint.
- MCP-based financial tools.
- Health endpoint for service checks.
- Structured scoring results for the AgentBeats evaluation and leaderboard workflow.

## Evaluation Metrics

The Green Agent evaluates each Purple Agent answer against the corresponding benchmark reference answer. Its LLM judge is intended to accommodate semantically equivalent phrasing, numeric equivalence, and task-specific interpretation rather than relying only on exact string matching.

The primary aggregate metric is **accuracy**:

```text
accuracy = correct_tasks / total_tasks
```

The result payload can report `accuracy`, `correct_tasks`, and `total_tasks`, together with any additional task-level details emitted by the current evaluator. The leaderboard reads the structured assessment artifacts/results returned through the A2A integration. A valid A2A response alone is not sufficient if the assessment results are not included in the expected output structure.

## References

- **Vals.ai Benchmark:** https://www.vals.ai/benchmarks/finance_agent
- **Vals.ai GitHub:** https://github.com/vals-ai/finance-agent
- **AgentBeats:** https://github.com/agentbeats
- **AgentBeats Documentation:** https://docs.agentbeats.org/
- **MCP Specification:** https://modelcontextprotocol.io/
- **A2A Protocol:** https://a2a-protocol.org/
- **SEC EDGAR APIs:** https://www.sec.gov/search-filings/edgar-application-programming-interfaces

## Contributing

This project originated as a UC Berkeley Agentic AI class project and is intended to support reproducible financial-agent evaluation. Contributions to the Green Agent should preserve the A2A assessment contract, MCP tool compatibility, and consistent scoring behavior. Purple Agent improvements can be developed and evaluated independently.

## License

MIT License — see the repository's LICENSE file for details.

## Acknowledgments

- **Vals.ai** for the original Finance-Agent-Benchmark.
- **UC Berkeley / AgentBeats** for the agent evaluation ecosystem.
- **Course instructors and collaborators** for guidance and feedback.

## Contact

- Fabio — inphlection@gmail.com
- Kiarash — kiarash996@gmail.com
- Milad — milad.eslamzadeh@teleperformance.com

---

**Status:** AgentBeats-integrated Green/Purple architecture; containerized benchmark and participant images available.