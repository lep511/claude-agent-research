# =============================================================================
# Plan Mode Helper Functions
# =============================================================================
# These utilities handle the various ways an agent might output its plan.
# Since agents can output plans via direct text, Write tool, or Claude's
# internal plan directory, we need robust extraction from multiple sources.

import glob as glob_module
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def extract_plan_from_xml(text: str | None, min_length: int = 50) -> str | None:
    """
    Extract content between <plan> tags from text.

    Args:
        text: The text to search for plan content
        min_length: Minimum character count for valid plan (prevents empty matches)

    Returns:
        Extracted plan content, or None if not found/too short
    """
    if not text:
        return None
    match = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if len(extracted) > min_length:
            return extracted
    return None


def extract_plan_from_messages(
    plan_content: list[str], min_fallback_length: int = 500
) -> tuple[str | None, str | None]:
    """
    Try to extract plan from captured message stream content.

    Args:
        plan_content: List of text blocks captured during streaming
        min_fallback_length: Minimum length for fallback (no XML tags)

    Returns:
        Tuple of (plan_text, source_description)
    """
    combined_text = "\n\n".join(plan_content)

    # First try: XML tags
    plan = extract_plan_from_xml(combined_text)
    if plan:
        return plan, "message stream"

    # Fallback: Use raw content if substantial
    if len(combined_text.strip()) > min_fallback_length:
        return combined_text.strip(), "full message content (fallback)"

    return None, None


def extract_plan_from_write_tool(
    write_contents: list[str], min_fallback_length: int = 500
) -> tuple[str | None, str | None]:
    """
    Try to extract plan from captured Write tool calls.

    Args:
        write_contents: List of content strings from Write tool calls
        min_fallback_length: Minimum length for fallback (no XML tags)

    Returns:
        Tuple of (plan_text, source_description)
    """
    for content in write_contents:
        # Try XML extraction first
        plan = extract_plan_from_xml(content)
        if plan:
            return plan, "Write tool capture"

        # Fallback: substantial content without tags
        if content and len(content.strip()) > min_fallback_length:
            return content.strip(), "Write tool capture (no XML tags)"

    return None, None


def extract_plan_from_claude_dir(
    max_age_seconds: int = 300, min_fallback_length: int = 500
) -> tuple[str | None, str | None]:
    """
    Check Claude's internal plan directory for recently created plans.

    Args:
        max_age_seconds: Maximum age of plan file to consider (default: 5 minutes)
        min_fallback_length: Minimum length for fallback (no XML tags)

    Returns:
        Tuple of (plan_text, source_description)
    """
    claude_plans_dir = os.path.expanduser("~/.claude/plans")

    if not os.path.exists(claude_plans_dir):
        return None, None

    # Find most recent plan file
    plan_files = sorted(
        glob_module.glob(os.path.join(claude_plans_dir, "*.md")),
        key=os.path.getmtime,
        reverse=True,
    )

    if not plan_files:
        return None, None

    most_recent = plan_files[0]
    file_age = datetime.now().timestamp() - os.path.getmtime(most_recent)

    if file_age > max_age_seconds:
        return None, None

    with open(most_recent) as f:
        content = f.read()

    filename = os.path.basename(most_recent)

    # Try XML extraction first
    plan = extract_plan_from_xml(content)
    if plan:
        return plan, f"Claude plan file ({filename})"

    # Fallback: substantial content without tags
    if len(content.strip()) > min_fallback_length:
        return content.strip(), f"Claude plan file ({filename}, no XML tags)"

    return None, None


def save_plan_to_file(
    plan_content: str,
    plan_source: str,
    model_name: str,
    prompt_summary: str,
    output_dir: str,
    title: str = "Agent Plan: Engineering Restructure for AI Focus",
) -> Path:
    """
    Save extracted plan to a timestamped markdown file.

    Args:
        plan_content: The plan text to save
        plan_source: Description of where plan was extracted from
        model_name: The model used to generate the plan
        prompt_summary: Brief description of the original prompt
        output_dir: Directory to save plan files
        title: Title for the plan document

    Returns:
        Path to the saved plan file
    """
    plans_dir = Path(output_dir)
    plans_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_file = plans_dir / f"plan_{timestamp}.md"

    with open(plan_file, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Prompt:** {prompt_summary}\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Plan Source:** {plan_source}\n\n")
        f.write("---\n\n")
        f.write(plan_content)
        f.write("\n\n---\n\n")
        f.write("*This plan was generated in plan mode and has not been executed.*\n")

    return plan_file


def capture_message_content(
    messages: Any,
) -> tuple[list[Any], list[Any], list[Any]]:
    """
    Process all messages from the agent stream and capture relevant plan content.

    This function extracts content from three potential sources:
    1. Text blocks in message content
    2. Write tool call parameters
    3. Final result attribute

    Args:
        messages: List of message objects from the agent stream
    """
    plan_content = []
    write_tool_content = []
    write_tool_paths = []

    for msg in messages:
        # Source 1: Text blocks from message content
        if hasattr(msg, "content"):
            for block in msg.content:
                if hasattr(block, "text"):
                    plan_content.append(block.text)

                # Source 2: Write tool calls
                if hasattr(block, "type") and block.type == "tool_use":
                    if hasattr(block, "name") and block.name == "Write":
                        if hasattr(block, "input") and isinstance(block.input, dict):
                            if "content" in block.input:
                                write_tool_content.append(block.input["content"])
                            if "file_path" in block.input:
                                write_tool_paths.append(block.input["file_path"])

        # Source 3: Final result
        if hasattr(msg, "result") and msg.result:
            plan_content.append(msg.result)

    return (plan_content, write_tool_content, write_tool_paths)


def extract_save_plan(
    plan_content: list[str],
    write_tool_content: list[str],
    write_tool_paths: list[str],
    prompt_summary: str,
    output_dir: str,
    model_name: str = "unknown",
) -> None:
    """
    Try multiple sources in priority order to find the plan content.
    This handles different agent behaviors robustly.
    """
    final_plan = None
    plan_source = None

    # Priority 1: Message stream (preferred - direct from agent response)
    final_plan, plan_source = extract_plan_from_messages(plan_content)

    # Priority 2: Write tool captures (if agent saved despite instructions)
    if not final_plan and write_tool_content:
        final_plan, plan_source = extract_plan_from_write_tool(write_tool_content)

    # Priority 3: Claude's internal plan directory (safety net)
    if not final_plan:
        final_plan, plan_source = extract_plan_from_claude_dir()

    # Report results
    if final_plan:
        print(f"✅ Plan extracted from: {plan_source}")
        print(f"   Plan length: {len(final_plan):,} characters")

        # Save to file
        plan_file = save_plan_to_file(
            plan_content=final_plan,
            plan_source=plan_source,
            model_name=model_name,
            output_dir=output_dir,
            prompt_summary=prompt_summary,
        )
        print(f"\n📁 Plan saved to: {plan_file}")
    else:
        error_msg = "Could not extract plan content from any source!\n"
        error_msg += "   Sources checked: message stream, Write tool, ~/.claude/plans/"
        if write_tool_paths:
            error_msg += f"\n   Write tool attempted to save to: {write_tool_paths}"
        print(f"❌ ERROR: {error_msg}")
        raise RuntimeError(f"Plan extraction failed: {error_msg}")