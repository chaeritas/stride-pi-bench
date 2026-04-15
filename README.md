# STRIDE Pi-Bench Agent

STRIDE XAI-optimized Purple Agent for [Pi-Bench](https://github.com/Jyoti-Ranjan-Das845/pi-bench) (AgentBeats Competition, UC Berkeley RDI).

## Results

| Domain | Overall | Compliance |
|--------|---------|------------|
| Helpdesk | **85.4%** | 37.5% (12/32) |
| Retail | **79.1%** | 15.4% (2/13) |
| FINRA | **72.0%** | 0.0% (0/26) |
| **Total** | **78.8%** | **19.7% (14/71)** |

**2.4x improvement** over baseline (33.1%) using GPT-4o-mini.

## Method

This agent uses **STRIDE XAI** to systematically optimize agent configuration parameters. Instead of manual prompt engineering, STRIDE prescribes optimal per-scenario configurations based on data-driven analysis.

## About

- **STRIDE XAI** by [Chaestro Inc.](https://chaestro.com)
- Patent: PCT/KR2026/004478
- Competition: [AgentBeats](https://agentbeats.dev) by [UC Berkeley RDI](https://rdi.berkeley.edu)

## License

MIT (Agent wrapper code only. STRIDE core engine is proprietary.)
