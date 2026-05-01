#!/usr/bin/env python3
"""Simple VAD test"""

import asyncio
import base64
import json
import wave
import websockets


async def test_simple():
    uri = "ws://localhost:28787/api-ws/v1/realtime"

    async with websockets.connect(uri) as ws:
        # Wait for session.created
        response = await ws.recv()
        data = json.loads(response)
        print(f"1. {data['type']}: {data['session']['id']}")

        # Enable VAD
        await ws.send(
            json.dumps(
                {
                    "event_id": "e1",
                    "type": "session.update",
                    "session": {
                        "input_audio_format": "pcm",
                        "sample_rate": 16000,
                        "input_audio_transcription": {"language": "zh"},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.3,
                            "silence_duration_ms": 500,
                        },
                    },
                }
            )
        )

        response = await ws.recv()
        print(f"2. {json.loads(response)['type']}")

        # Send a small audio chunk (1 second)
        with wave.open("test.wav", "rb") as f:
            # Read only first 1 second (16000 samples * 2 bytes)
            audio_data = f.readframes(16000)
            audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        await ws.send(
            json.dumps({"event_id": "e2", "type": "input_audio_buffer.append", "audio": audio_b64})
        )

        print("3. Sent 1 second audio")

        # Try to receive events
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(response)
                print(f"   Event: {data['type']}")
                if data["type"] == "input_audio_buffer.speech_started":
                    print("   ✓ VAD detected speech start!")
        except asyncio.TimeoutError:
            print("   (no more events)")

        # Finish
        await ws.send(json.dumps({"event_id": "e3", "type": "session.finish"}))

        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"   Event: {data['type']}")
                if data["type"] == "session.finished":
                    break
        except asyncio.TimeoutError:
            print("   Timeout waiting for finish")

        except asyncio.TimeoutError:
            print("   Timeout waiting for finish")

if __name__ == "__main__":
    asyncio.run(test_simple())
