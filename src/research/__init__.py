import os
from dotenv import load_dotenv
from research_agent.agent import send_query as resarch_send_query
from chief_of_staff_agent.agent import send_query as cs_send_query
from utils.agent_visualizer import visualize_conversation, print_markdown_result
from utils.plan_mode_helper_functions import capture_message_content, extract_save_plan
import asyncio

# Load .env (model IDs, region) from this module folder
load_dotenv()

# Use Amazon Bedrock as the model provider.
# With this set, the SDK reads ANTHROPIC_MODEL from the environment — so we never
# hardcode a model in code, and you can switch models from .env alone.
os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
DEFAULT_MODEL = os.getenv('ANTHROPIC_MODEL', 'NOT SET')
SMALL_FAST_MODEL = os.getenv('ANTHROPIC_SMALL_FAST_MODEL', 'NOT SET')
AWS_REGION = os.getenv('AWS_REGION', 'NOT SET')

print("✅ Provider: Amazon Bedrock")
print(f"   Model:            {DEFAULT_MODEL}")
print(f"   Small/fast model: {SMALL_FAST_MODEL}")
print(f"   AWS region:       {AWS_REGION}")

def main() -> None:
    # # CHART EXAMPLE

    # prompt = "Analyze the chart in research_agent/projects_claude.png"
    # prompt_test = "What is the Claude Code SDK? Only do one websearch and be concise"

    # result, messages = asyncio.run(resarch_send_query(
    #     prompt=prompt_test,
    #     model=DEFAULT_MODEL
    # ))
    # print_markdown_result(messages)
    # visualize_conversation(messages)

    # # CHIEF EXAMPLE
    # prompt = "What's our current runway?"
    # result, messages = asyncio.run(cs_send_query(
    #     prompt=prompt,
    #     model=DEFAULT_MODEL
    # ))
    # print_markdown_result(messages)

    # # PLAN MODE EXAMPLE
#     PLAN_PROMPT = """Restructure our engineering team for AI focus.

# **CONTEXT (from CLAUDE.md):**
# You are the Chief of Staff for TechStart Inc, a 50-person B2B SaaS startup.
# - Current engineering team: 25 people (Backend: 12, Frontend: 8, DevOps: 5)

# **OUTPUT INSTRUCTIONS:**

# 1. **DO NOT use the Write tool** - Output your plan directly in your response text
# 2. **Wrap your plan inside `<plan> </plan>` XML tags**

# **Required Format:**
# <plan>
# [A simple restructuring plan: proposed team structure and key hiring recommendations]
# </plan>

# Keep it brief and high-level. Do NOT ask clarifying questions."""

#     result, messages = asyncio.run(cs_send_query(
#         prompt=PLAN_PROMPT,
#         model=DEFAULT_MODEL,
#         permission_mode="plan",
#     ))
#     print_markdown_result(messages)
#     plan_content = []  # Text from message stream
#     write_tool_content = []  # Content from Write tool calls
#     write_tool_paths = []  # Paths from Write tool calls

#     # Capture content from this message
#     pc, wtc, wtp = capture_message_content(messages)
#     extract_save_plan(
#         plan_content=pc, 
#         write_tool_content=wtc, 
#         write_tool_paths=wtp, 
#         prompt_summary="Restructure our engineering team for AI focus.",
#         output_dir="plans",
#         model_name=DEFAULT_MODEL
#     )

    # # User types: "/slash-command-test this is a test"
    # # -> behind the scenes EXPANDS to the prompt in .claude/commands/slash-command-test.md
    # # In this case the expanded prompt says to simply reverse the sentence word wise
    # prompt = "/slash-command-test this is a test"
    # result, messages = asyncio.run(cs_send_query(
    #     prompt=prompt,
    #     model=DEFAULT_MODEL
    # ))
    # print_markdown_result(messages)

    # prompt = "Create a quick Q2 financial forecast report with our current burn rate and runway projections. Save it to our /output_reports folder."
    prompt = "Should we hire 5 engineers? Analyze the financial impact."
    result, messages = asyncio.run(cs_send_query(
        prompt=prompt,
        model=DEFAULT_MODEL
    ))
    print_markdown_result(messages)