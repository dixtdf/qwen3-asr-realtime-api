#!/usr/bin/env python3
"""
Demo Server for Qwen3-ASR-Realtime

Features:
1. Serves static files (index.html)
2. Proxies WebSocket connections to DashScope API (handles auth headers)
3. Proxies WebSocket connections to local ASR server

Usage:
    python demo/server.py [--port 7860] [--asr-server ws://localhost:28787]

Then open http://localhost:7860 in your browser.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import aiohttp
from aiohttp import WSMsgType, web

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Directory containing this script
DEMO_DIR = Path(__file__).parent


async def index_handler(request: web.Request) -> web.Response:
    """Serve index.html"""
    return web.FileResponse(DEMO_DIR / "index.html")


async def proxy_to_dashscope(request: web.Request) -> web.WebSocketResponse:
    """
    Proxy WebSocket connection to DashScope API.

    Browser WebSocket API cannot send custom headers, so this endpoint
    acts as a proxy that handles authentication server-side.

    Query params:
        api_key: DashScope API Key (required)
        model: Model name (default: qwen3-asr-flash-realtime)
        region: cn or intl (default: cn)
    """
    api_key = request.query.get("api_key")
    model = request.query.get("model", "qwen3-asr-flash-realtime")
    region = request.query.get("region", "cn")

    if not api_key:
        return web.Response(status=400, text="Missing api_key parameter")

    # Determine DashScope endpoint
    if region == "intl":
        dashscope_url = f"wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model={model}"
    else:
        dashscope_url = f"wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model={model}"

    logger.info(f"Proxying to DashScope: {dashscope_url}")

    # Accept WebSocket from browser
    ws_browser = web.WebSocketResponse()
    await ws_browser.prepare(request)

    try:
        # Connect to DashScope with auth headers
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                dashscope_url,
                headers={"Authorization": f"Bearer {api_key}", "OpenAI-Beta": "realtime=v1"},
            ) as ws_upstream:

                async def forward_to_upstream():
                    """Forward messages from browser to DashScope"""
                    try:
                        async for msg in ws_browser:
                            if msg.type == WSMsgType.TEXT:
                                await ws_upstream.send_str(msg.data)
                            elif msg.type == WSMsgType.BINARY:
                                await ws_upstream.send_bytes(msg.data)
                            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                                break
                    except Exception as e:
                        logger.error(f"Error forwarding to upstream: {e}")

                async def forward_to_browser():
                    """Forward messages from DashScope to browser"""
                    try:
                        async for msg in ws_upstream:
                            if msg.type == WSMsgType.TEXT:
                                await ws_browser.send_str(msg.data)
                            elif msg.type == WSMsgType.BINARY:
                                await ws_browser.send_bytes(msg.data)
                            elif msg.type == WSMsgType.ERROR:
                                logger.error(f"Upstream error: {ws_upstream.exception()}")
                                break
                            elif msg.type == WSMsgType.CLOSED:
                                break
                    except Exception as e:
                        logger.error(f"Error forwarding to browser: {e}")

                # Run both directions concurrently
                await asyncio.gather(
                    forward_to_upstream(), forward_to_browser(), return_exceptions=True
                )

    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to DashScope: {e}")
        if not ws_browser.closed:
            await ws_browser.send_json(
                {
                    "type": "error",
                    "error": {
                        "type": "proxy_error",
                        "message": f"Failed to connect to DashScope API: {str(e)}",
                    },
                }
            )
    except Exception as e:
        logger.error(f"Proxy error: {e}")
    finally:
        if not ws_browser.closed:
            await ws_browser.close()

    return ws_browser


async def proxy_to_local_asr(request: web.Request) -> web.WebSocketResponse:
    """
    Proxy WebSocket connection to local ASR server.

    This allows the demo page to connect to both the demo server
    and the local ASR server through the same origin.

    Query params:
        url: Local ASR server URL (default from app config)
    """
    local_url = request.query.get("url", request.app["asr_server_url"])

    logger.info(f"Proxying to local ASR: {local_url}")

    # Accept WebSocket from browser
    ws_browser = web.WebSocketResponse()
    await ws_browser.prepare(request)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(local_url) as ws_upstream:

                async def forward_to_upstream():
                    try:
                        async for msg in ws_browser:
                            if msg.type == WSMsgType.TEXT:
                                await ws_upstream.send_str(msg.data)
                            elif msg.type == WSMsgType.BINARY:
                                await ws_upstream.send_bytes(msg.data)
                            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                                break
                    except Exception as e:
                        logger.error(f"Error forwarding to local ASR: {e}")

                async def forward_to_browser():
                    try:
                        async for msg in ws_upstream:
                            if msg.type == WSMsgType.TEXT:
                                await ws_browser.send_str(msg.data)
                            elif msg.type == WSMsgType.BINARY:
                                await ws_browser.send_bytes(msg.data)
                            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSED):
                                break
                    except Exception as e:
                        logger.error(f"Error forwarding from local ASR: {e}")

                await asyncio.gather(
                    forward_to_upstream(), forward_to_browser(), return_exceptions=True
                )

    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to local ASR: {e}")
        if not ws_browser.closed:
            await ws_browser.send_json(
                {
                    "type": "error",
                    "error": {
                        "type": "proxy_error",
                        "message": f"Failed to connect to local ASR server: {str(e)}",
                    },
                }
            )
    except Exception as e:
        logger.error(f"Proxy error: {e}")
    finally:
        if not ws_browser.closed:
            await ws_browser.close()

    return ws_browser


async def config_handler(request: web.Request) -> web.Response:
    """Return server configuration for the demo page"""
    return web.json_response(
        {
            "asr_server_url": request.app["asr_server_url"],
            "demo_server": True,
            "proxy_endpoints": {"dashscope": "/ws/proxy/dashscope", "local": "/ws/proxy/local"},
        }
    )


def create_app(asr_server_url: str) -> web.Application:
    """Create the aiohttp application"""
    app = web.Application()
    app["asr_server_url"] = asr_server_url

    # Routes
    app.router.add_get("/", index_handler)
    app.router.add_get("/config", config_handler)
    app.router.add_get("/ws/proxy/dashscope", proxy_to_dashscope)
    app.router.add_get("/ws/proxy/local", proxy_to_local_asr)

    # Static files (for any additional assets)
    app.router.add_static("/static/", DEMO_DIR, name="static")

    return app


def main():
    parser = argparse.ArgumentParser(description="Demo server for Qwen3-ASR-Realtime")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument(
        "--asr-server",
        default="ws://localhost:28787/api-ws/v1/realtime",
        help="Local ASR server WebSocket URL",
    )
    args = parser.parse_args()

    app = create_app(args.asr_server)

    logger.info("=" * 60)
    logger.info("Qwen3-ASR Demo Server")
    logger.info("=" * 60)
    logger.info(f"Demo page: http://{args.host}:{args.port}/")
    logger.info(f"Local ASR proxy: ws://{args.host}:{args.port}/ws/proxy/local")
    logger.info(f"DashScope proxy: ws://{args.host}:{args.port}/ws/proxy/dashscope")
    logger.info(f"Upstream ASR server: {args.asr_server}")
    logger.info("=" * 60)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
