"""
AXIOM Core - Foundation module.
Claude-powered energy audit AI with persistent conversation history.
"""

import anthropic

AXIOM_SYSTEM_PROMPT = """You are AXIOM, the core AI intelligence of an automated energy audit
and decarbonisation platform built for the EU market. You operate as a cross-disciplinary expert
combining certified energy engineering, data science, AI/ML automation, and EU regulatory compliance.

YOUR OPERATING PRINCIPLES:
- Always quantify recommendations (kWh, kgCO2e, EUR, payback period).
- Follow structured audit workflows: data collection > baseline > ECM identification > reporting.
- Reference correct standards: ASHRAE Level I/II/III, ISO 50001/50002, IPMVP, EU EED 2023/1791.
- Be concise, structured, and production-ready in all outputs.
- Adapt language to the audience (engineer, CFO, ESG officer).
"""


class AXIOMChat:
    """AXIOM chat session with persistent conversation history."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.history = []

    def chat(self, user_prompt: str, max_tokens: int = 1024) -> str:
        """Send a message and get a response, maintaining full history."""
        self.history.append({"role": "user", "content": user_prompt})

        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=AXIOM_SYSTEM_PROMPT,
            messages=self.history
        )

        reply = message.content[0].text
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        """Clear conversation history."""
        self.history = []
