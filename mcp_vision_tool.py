"""
mcp_vision_tool.py
------------------
A custom MCP tool server that gives Mimi true self-vision.

Instead of returning just a file path, the `screenshot_and_see` tool:
  1. Uses Playwright to navigate and screenshot a URL (or the current page)
  2. Base64-encodes the PNG
  3. Returns it as a multimodal content block so Mimi can SEE what she captured

Install dependencies:
    pip install mcp playwright
    playwright install chromium

Run:
    python mcp_vision_tool.py

Then register it in your MCP config (e.g. mcp_config.json):
    {
      "mcpServers": {
        "vision": {
          "command": "python",
          "args": ["/path/to/mcp_vision_tool.py"]
        }
      }
    }
"""

import asyncio
import base64
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# MCP SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    CallToolResult,
)

# Playwright
from playwright.async_api import async_playwright

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path(os.environ.get("SCREENSHOT_DIR", Path.home() / "mimi-screenshots"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MCP Server setup ──────────────────────────────────────────────────────────
server = Server("mimi-vision")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="screenshot_and_see",
            description=(
                "Navigate to a URL (or use the currently open browser page), take a "
                "screenshot, and return the image directly so you can visually analyze "
                "what is on screen. Use this whenever you need to see a webpage, chart, "
                "dashboard, or any visual content. The image is returned to your vision "
                "system automatically — you do NOT need to ask the user to send it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to before taking the screenshot. Leave empty to screenshot the current page."
                    },
                    "wait_seconds": {
                        "type": "number",
                        "description": "Seconds to wait after page load before screenshotting. Default 2. Increase for JS-heavy pages like TradingView.",
                        "default": 2
                    },
                    "full_page": {
                        "type": "boolean",
                        "description": "Capture the full scrollable page. Default false (viewport only).",
                        "default": False
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Optional custom file path to save the PNG. Auto-generated if omitted."
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="read_image_and_see",
            description=(
                "Read an existing image file from disk and return it to your vision system "
                "so you can analyze it. Use this to look at screenshots or images that were "
                "already saved to disk."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the image file (PNG, JPG, WEBP, GIF)."
                    }
                },
                "required": ["file_path"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    if name == "screenshot_and_see":
        return await handle_screenshot_and_see(arguments)
    elif name == "read_image_and_see":
        return await handle_read_image_and_see(arguments)
    else:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            isError=True
        )


async def handle_screenshot_and_see(args: dict) -> CallToolResult:
    """Take a Playwright screenshot and return it as a vision-accessible image block."""
    url = args.get("url", "").strip()
    wait_seconds = float(args.get("wait_seconds", 2))
    full_page = bool(args.get("full_page", False))

    # Determine save path
    save_path = args.get("save_path", "").strip()
    if not save_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = str(OUTPUT_DIR / f"screenshot_{timestamp}.png")

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=1,
            )
            page = await context.new_page()

            if url:
                await page.goto(url, wait_until="networkidle", timeout=30000)
            
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

            await page.screenshot(path=save_path, full_page=full_page)
            await browser.close()

        # Read and encode the saved screenshot
        with open(save_path, "rb") as f:
            image_bytes = f.read()

        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Screenshot captured: {save_path}\nURL: {url or '(current page)'}\nSize: {len(image_bytes):,} bytes"
                ),
                ImageContent(
                    type="image",
                    data=image_b64,
                    mimeType="image/png"
                )
            ]
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Screenshot failed: {str(e)}")],
            isError=True
        )


async def handle_read_image_and_see(args: dict) -> CallToolResult:
    """Read an image file from disk and return it as a vision-accessible image block."""
    file_path = args.get("file_path", "").strip()

    if not file_path:
        return CallToolResult(
            content=[TextContent(type="text", text="Error: file_path is required.")],
            isError=True
        )

    path = Path(file_path)
    if not path.exists():
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: File not found: {file_path}")],
            isError=True
        )

    # Determine MIME type from extension
    ext = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_map.get(ext, "image/png")

    try:
        with open(path, "rb") as f:
            image_bytes = f.read()

        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Image loaded: {file_path}\nSize: {len(image_bytes):,} bytes"
                ),
                ImageContent(
                    type="image",
                    data=image_b64,
                    mimeType=mime_type
                )
            ]
        )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Failed to read image: {str(e)}")],
            isError=True
        )


# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
