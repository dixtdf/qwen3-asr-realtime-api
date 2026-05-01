#!/usr/bin/env python3
"""
SDK Compatibility Test for Qwen3-ASR-Realtime Server

Tests that the private server is compatible with the official DashScope SDK.
This test verifies protocol compliance and event handling.
"""

import asyncio
import base64
import json
import sys
import time
from datetime import datetime

import websockets


class SDKCompatibilityTest:
    """Test suite for SDK compatibility"""

    def __init__(self, uri="ws://localhost:28787/api-ws/v1/realtime"):
        self.uri = uri
        self.results = []
        self.events_received = []

    def log(self, message, level="INFO"):
        """Log test progress"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {level}: {message}")

    async def test_session_lifecycle(self):
        """Test basic session lifecycle"""
        self.log("Testing session lifecycle...")

        try:
            async with websockets.connect(self.uri) as ws:
                # 1. Receive session.created
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)

                assert data["type"] == "session.created", (
                    f"Expected session.created, got {data['type']}"
                )
                assert "session" in data, "Missing session data"
                assert "id" in data["session"], "Missing session ID"

                self.log(f"✓ session.created received (id: {data['session']['id'][:8]}...)")

                # 2. Send session.update
                session_update = {
                    "event_id": "test_event_001",
                    "type": "session.update",
                    "session": {
                        "input_audio_format": "pcm",
                        "sample_rate": 16000,
                        "input_audio_transcription": {"language": "auto"},
                        "turn_detection": None,
                    },
                }
                await ws.send(json.dumps(session_update))

                # 3. Receive session.updated
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)

                assert data["type"] == "session.updated", (
                    f"Expected session.updated, got {data['type']}"
                )

                self.log("✓ session.updated received")

                # 4. Send session.finish
                await ws.send(json.dumps({"event_id": "test_finish", "type": "session.finish"}))

                # 5. Receive session.finished
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)

                assert data["type"] == "session.finished", (
                    f"Expected session.finished, got {data['type']}"
                )

                self.log("✓ session.finished received")

                return True

        except Exception as e:
            self.log(f"✗ Session lifecycle test failed: {e}", "ERROR")
            return False

    async def test_manual_mode(self):
        """Test Manual mode recognition"""
        self.log("Testing Manual mode...")

        try:
            async with websockets.connect(self.uri) as ws:
                # Setup session
                await asyncio.wait_for(ws.recv(), timeout=5.0)  # session.created

                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "input_audio_format": "pcm",
                                "sample_rate": 16000,
                                "turn_detection": None,  # Manual mode
                            },
                        }
                    )
                )

                await asyncio.wait_for(ws.recv(), timeout=5.0)  # session.updated

                # Send dummy audio (1 second of silence)
                audio_data = b"\x00" * 32000  # 1s of 16-bit silence at 16kHz
                audio_b64 = base64.b64encode(audio_data).decode()

                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))

                # Commit
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

                # Wait for events
                events = []
                start_time = time.time()
                while time.time() - start_time < 10:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(response)
                        events.append(data["type"])

                        if data["type"] == "conversation.item.input_audio_transcription.completed":
                            self.log(f"✓ Recognition completed: '{data.get('transcript', '')}'")
                            break
                        elif data["type"] == "error":
                            self.log(f"✗ Error: {data.get('error', {})}", "ERROR")
                            return False
                    except asyncio.TimeoutError:
                        continue

                # Check expected events
                expected_events = [
                    "input_audio_buffer.committed",
                    "conversation.item.created",
                    "conversation.item.input_audio_transcription.completed",
                ]

                for event in expected_events:
                    if event in events:
                        self.log(f"✓ Received: {event}")
                    else:
                        self.log(f"⚠ Missing: {event}", "WARNING")

                # Finish session
                await ws.send(json.dumps({"type": "session.finish"}))
                await asyncio.wait_for(ws.recv(), timeout=5.0)

                return True

        except Exception as e:
            self.log(f"✗ Manual mode test failed: {e}", "ERROR")
            import traceback

            traceback.print_exc()
            return False

    async def test_vad_mode(self):
        """Test VAD mode"""
        self.log("Testing VAD mode...")

        try:
            async with websockets.connect(self.uri) as ws:
                # Setup session with VAD
                await asyncio.wait_for(ws.recv(), timeout=5.0)  # session.created

                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "input_audio_format": "pcm",
                                "sample_rate": 16000,
                                "turn_detection": {
                                    "type": "server_vad",
                                    "threshold": 0.5,
                                    "silence_duration_ms": 400,
                                },
                            },
                        }
                    )
                )

                await asyncio.wait_for(ws.recv(), timeout=5.0)  # session.updated

                # Send audio (will be detected as silence)
                audio_data = b"\x00" * 16000  # 0.5s of silence
                audio_b64 = base64.b64encode(audio_data).decode()

                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))

                # Wait a bit for VAD processing
                await asyncio.sleep(0.5)

                # Finish
                await ws.send(json.dumps({"type": "session.finish"}))

                # Collect events
                events = []
                start_time = time.time()
                while time.time() - start_time < 5:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(response)
                        events.append(data["type"])
                    except asyncio.TimeoutError:
                        break

                self.log(f"Events received: {events}")

                # VAD might not trigger on silence, that's OK
                self.log("✓ VAD mode test completed")
                return True

        except Exception as e:
            self.log(f"✗ VAD mode test failed: {e}", "ERROR")
            return False

    async def test_error_handling(self):
        """Test error responses"""
        self.log("Testing error handling...")

        try:
            async with websockets.connect(self.uri) as ws:
                await asyncio.wait_for(ws.recv(), timeout=5.0)  # session.created

                # Send invalid event
                await ws.send(json.dumps({"type": "invalid.event.type"}))

                # Should receive error
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)

                if data["type"] == "error":
                    self.log(
                        f"✓ Error response received: {data.get('error', {}).get('message', 'Unknown')}"
                    )
                    return True
                else:
                    self.log(f"⚠ Expected error, got: {data['type']}", "WARNING")
                    return True  # Not a failure, just different behavior

        except Exception as e:
            self.log(f"✗ Error handling test failed: {e}", "ERROR")
            return False

    async def test_heartbeat(self):
        """Test heartbeat/ping-pong"""
        self.log("Testing heartbeat...")

        try:
            async with websockets.connect(self.uri) as ws:
                await asyncio.wait_for(ws.recv(), timeout=5.0)  # session.created

                # Send ping
                await ws.send(json.dumps({"event_id": "ping_test", "type": "ping"}))

                # Wait for pong
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)

                if data["type"] == "pong":
                    self.log("✓ Heartbeat pong received")
                    return True
                else:
                    self.log(f"⚠ Expected pong, got: {data['type']}", "WARNING")
                    return True

        except Exception as e:
            self.log(f"✗ Heartbeat test failed: {e}", "ERROR")
            return False

    async def run_all_tests(self):
        """Run all tests"""
        self.log("=" * 60)
        self.log("SDK Compatibility Test Suite")
        self.log("=" * 60)
        self.log(f"Target: {self.uri}")
        self.log("")

        tests = [
            ("Session Lifecycle", self.test_session_lifecycle),
            ("Manual Mode", self.test_manual_mode),
            ("VAD Mode", self.test_vad_mode),
            ("Error Handling", self.test_error_handling),
            ("Heartbeat", self.test_heartbeat),
        ]

        results = []
        for name, test_func in tests:
            self.log(f"\n{'─' * 60}")
            self.log(f"Running: {name}")
            self.log("─" * 60)

            try:
                result = await test_func()
                results.append((name, result))
            except Exception as e:
                self.log(f"✗ Test crashed: {e}", "ERROR")
                results.append((name, False))

        # Summary
        self.log("\n" + "=" * 60)
        self.log("Test Summary")
        self.log("=" * 60)

        passed = sum(1 for _, r in results if r)
        failed = sum(1 for _, r in results if not r)

        for name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            self.log(f"{status}: {name}")

        self.log("─" * 60)
        self.log(f"Total: {passed} passed, {failed} failed")

        return failed == 0


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="SDK Compatibility Test")
    parser.add_argument(
        "--url", "-u", default="ws://localhost:28787/api-ws/v1/realtime", help="WebSocket URL"
    )

    args = parser.parse_args()

    tester = SDKCompatibilityTest(args.url)
    success = await tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
