#!/usr/bin/env python3
"""VAD mode with ASR test"""

import asyncio
import base64
import json
import wave
import websockets


async def test_vad_asr():
    uri = "ws://localhost:28787/api-ws/v1/realtime"

    async with websockets.connect(uri) as ws:
        # session.created
        response = await ws.recv()
        print(f"1. {json.loads(response)['type']}")

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

        await ws.recv()
        print("2. VAD enabled")

        # Send 5 seconds audio in chunks
        with wave.open("test.wav", "rb") as f:
            for i in range(5):
                chunk = f.readframes(16000)  # 1 second
                if not chunk:
                    break
                await ws.send(
                    json.dumps(
                        {
                            "event_id": f"a{i}",
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("utf-8"),
                        }
                    )
                )
                await asyncio.sleep(0.3)

        print("3. Sent 5 seconds audio")

        # Collect events
        events = []
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=3.0)
                data = json.loads(response)
                events.append(data["type"])
                print(f"   → {data['type']}")
                if "text" in data:
                    print(f"      text: {data['text']}")
                if "stash" in data:
                    print(f"      stash: {data['stash']}")
                if "transcript" in data:
                    print(f"      FINAL: {data['transcript']}")
        except asyncio.TimeoutError:
            pass

        # Finish
        await ws.send(json.dumps({"event_id": "e2", "type": "session.finish"}))

        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)
                events.append(data["type"])
                print(f"   → {data['type']}")
                if "transcript" in data:
                    print(f"      FINAL: {data['transcript']}")
                if data["type"] == "session.finished":
                    break
        except asyncio.TimeoutError:
            print("   Timeout")

        print(f"\n✓ All events: {events}")


        print(f"\n✓ All events: {events}")

if __name__ == "__main__":
    asyncio.run(test_vad_asr())
