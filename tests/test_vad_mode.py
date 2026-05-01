#!/usr/bin/env python3
"""Test VAD mode"""

import asyncio
import base64
import json
import wave
import websockets


async def test_vad_mode():
    uri = "ws://localhost:28787/api-ws/v1/realtime"

    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as ws:
            print("Connected!")

            # Wait for session.created
            response = await ws.recv()
            data = json.loads(response)
            print(f"\n1. Received: {data['type']}")
            print(f"   Session ID: {data['session']['id']}")

            # Send session.update with VAD enabled
            session_update = {
                "event_id": "event_001",
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
            await ws.send(json.dumps(session_update))
            print("\n2. Sent: session.update (VAD enabled)")

            # Wait for session.updated
            response = await ws.recv()
            data = json.loads(response)
            print(f"3. Received: {data['type']}")
            print(f"   VAD: {data['session'].get('turn_detection')}")

            # Load and send audio
            with wave.open("test.wav", "rb") as wav_file:
                audio_data = wav_file.readframes(wav_file.getnframes())
                audio_b64 = base64.b64encode(audio_data).decode("utf-8")

                # Send in smaller chunks
                chunk_size = 16000  # 0.5 second
                chunks = [
                    audio_b64[i : i + chunk_size] for i in range(0, len(audio_b64), chunk_size)
                ]

                print(f"\n4. Sending audio in {len(chunks)} chunks...")

                for i, chunk in enumerate(chunks[:20]):  # Send first 10 seconds
                    audio_msg = {
                        "event_id": f"audio_{i:03d}",
                        "type": "input_audio_buffer.append",
                        "audio": chunk,
                    }
                    await ws.send(json.dumps(audio_msg))
                    await asyncio.sleep(0.3)  # Simulate real-time

                    # Receive events
                    try:
                        while True:
                            response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                            data = json.loads(response)
                            print(f"   Event: {data['type']}")
                            if "text" in data or "stash" in data:
                                print(f"      Text: {data.get('text', '')}{data.get('stash', '')}")
                            if "transcript" in data:
                                print(f"      Final: {data['transcript']}")
                    except asyncio.TimeoutError:
                        pass

                print("\n5. Audio sent, finishing session...")

                finish_msg = {"event_id": "event_finish", "type": "session.finish"}
                await ws.send(json.dumps(finish_msg))
                print("   Sent: session.finish")

                try:
                    while True:
                        response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        data = json.loads(response)
                        print(f"   Event: {data['type']}")
                        if data["type"] == "conversation.item.input_audio_transcription.completed":
                            print(f"      Final: {data.get('transcript', '')}")
                        if data["type"] == "session.finished":
                            break
                except asyncio.TimeoutError:
                    print("   Timeout waiting for results")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vad_mode())
