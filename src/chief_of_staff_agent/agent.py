"""
Chief of Staff Agent

An AI-powered Chief of Staff for TechStart Inc, built with the Claude Agent SDK.
This agent acts as an executive assistant capable of financial analysis,
recruiting support, and strategic decision-making by combining native tools,
custom Python scripts, and filesystem-based configuration.

Core Capabilities:
    - Subagent delegation via the Task tool to specialized agents
      (financial-analyst, recruiter) defined in .claude/agents/
    - Custom Python scripts executed via Bash for:
        * financial_forecast.py — advanced financial modeling
        * talent_scorer.py — candidate scoring algorithm
        * decision_matrix.py — strategic decision framework
    - Access to company financial data in the financial_data/ directory
    - Persistent project context loaded from CLAUDE.md
    - Slash commands expanded from .claude/commands/
    - Configurable output styles (e.g., executive, technical, board-report)
    - Hooks triggered via .claude/settings.local.json

Requirements:
    - setting_sources must include "project" for the SDK to load
      filesystem-based configuration (commands, subagents, CLAUDE.md, hooks).
      Without it, the agent runs in isolation mode with no project context.
"""

import asyncio
import json
import os
from collections.abc import Callable
from typing import Any, Literal
from dotenv import load_dotenv
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient


AGENT_SYSTEM_PROMPT = """You are the Chief of Staff for TechStart Inc, a 50-person startup.

Delegate financial questions to the financial-analyst subagent. Do not try to answer these questions yourself.
Apart from your tools and two subagents, you also have custom Python scripts in the scripts/ directory you can run with Bash:
- python scripts/financial_forecast.py: Advanced financial modeling
- python scripts/talent_scorer.py: Candidate scoring algorithm
- python scripts/decision_matrix.py: Strategic decision framework

You have access to company data in the financial_data/ directory.
"""


def get_activity_text(msg: Any) -> str | None:
    """Extract activity text from a message"""
    try:
        if "Assistant" in msg.__class__.__name__:
            if hasattr(msg, "content") and msg.content:
                first_content = msg.content[0] if isinstance(msg.content, list) else msg.content
                if hasattr(first_content, "name"):
                    return f"🤖 Using: {first_content.name}()"
            return "🤖 Thinking..."
        elif "User" in msg.__class__.__name__:
            return "✓ Tool completed"
    except (AttributeError, IndexError):
        pass
    return None


def print_activity(msg: Any) -> None:
    """Print activity to console"""
    activity = get_activity_text(msg)
    if activity:
        print(activity)


async def send_query(
    prompt: str,
    model: str,
    continue_conversation: bool = False,
    permission_mode: Literal["default", "plan", "acceptEdits"] = "default",
    output_style: str | None = None,
    activity_handler: Callable[[Any], None | Any] = print_activity,
) -> tuple[str | None, list[Any]]:
    """
    Send a query to the Chief of Staff agent with all features integrated.

    Args:
        prompt: The query to send (can include slash commands like /budget-impact)
        activity_handler: Callback for activity updates (default: print_activity)
        continue_conversation: Continue the previous conversation if True
        permission_mode: "default" (execute), "plan" (think only), or "acceptEdits"
        output_style: Override output style (e.g., "executive", "technical", "board-report")

    Returns:
        Tuple of (result, messages) - result is the final text, messages is the full conversation

    Features automatically included/leveraged:
        - Memory: CLAUDE.md context loaded from chief_of_staff/CLAUDE.md
        - Subagents: financial-analyst and recruiter via Task tool (defined in .claude/agents)
        - Custom scripts: Python scripts in tools/ via Bash
        - Slash commands: Expanded from .claude/commands/
        - Output styles: Custom output styles defined in .claude/output-styles
        - Hooks: Triggered based on settings.local.json, defined in .claude/hooks
    """

    # build options with optional output style
    settings = None
    if output_style:
        settings = json.dumps({"outputStyle": output_style})

    options = ClaudeAgentOptions(
        model=model,
        allowed_tools=[
            "Task",  # enables subagent delegation
            "Read",
            "Write",
            "Edit",
            "Bash",
            "WebSearch",
            "MultiEdit",
        ],
        continue_conversation=continue_conversation,
        system_prompt=AGENT_SYSTEM_PROMPT,
        permission_mode=permission_mode,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        settings=settings,
        # IMPORTANT: setting_sources must include "project" to load filesystem settings:
        # - Slash commands from .claude/commands/
        # - CLAUDE.md project instructions
        # - Subagent definitions from .claude/agents/
        # - Hooks from .claude/settings.local.json
        # Without this, the SDK operates in isolation mode with no filesystem settings loaded.
        setting_sources=["project", "local"],
    )

    result: str | None = None
    messages: list[Any] = []

    try:
        async with ClaudeSDKClient(options=options) as agent:
            await agent.query(prompt=prompt)
            async for msg in agent.receive_response():
                messages.append(msg)
                if asyncio.iscoroutinefunction(activity_handler):
                    await activity_handler(msg)
                else:
                    activity_handler(msg)

                if hasattr(msg, "result"):
                    result = msg.result
    except Exception as e:
        print(f"❌ Query error: {e}")
        raise

    return result, messages