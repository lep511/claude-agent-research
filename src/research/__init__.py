import os
from dotenv import load_dotenv
from research_agent.agent import send_query as resarch_send_query
from chief_of_staff_agent.agent import send_query as cs_send_query
from utils.agent_visualizer import visualize_conversation, print_markdown_result
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
    # prompt = "Analyze the chart in research_agent/projects_claude.png"
    # prompt_test = "What is the Claude Code SDK? Only do one websearch and be concise"

    # result, messages = asyncio.run(resarch_send_query(
    #     prompt=prompt_test,
    #     model=DEFAULT_MODEL
    # ))
    # print_markdown_result(messages)
    # visualize_conversation(messages)

    prompt = "What's our current runway?"
    prompt = "Tell me in two sentences about your writing output style."
    result, messages = asyncio.run(cs_send_query(
        prompt=prompt,
        model=DEFAULT_MODEL
    ))
    print_markdown_result(messages)

