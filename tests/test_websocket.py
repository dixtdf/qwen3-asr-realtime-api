#!/usr/bin/env python3
"""
Test script for Qwen3-ASR-Realtime Server
Compatible with Alibaba Cloud Qwen3-ASR-realtime API
"""

import asyncio
import base64
import json
import sys
import wave

import websockets


async def test_websocket_server():
    uri = "ws://localhost:28787/api-ws/v1/realtime"

    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as ws:
            print("Connected!")

            # 1. Wait for session.created
            response = await ws.recv()
            data = json.loads(response)
            print(f"\n1. Received: {data['type']}")
            print(f"   Session ID: {data['session']['id']}")
            print(f"   Model: {data['session']['model']}")

            # 2. Send session.update
            session_update = {
                "event_id": "event_001",
                "type": "session.update",
                "session": {
                    "input_audio_format": "pcm",
                    "sample_rate": 16000,
                    "input_audio_transcription": {"language": "Chinese"},
                    "turn_detection": None,
                },
            }
            await ws.send(json.dumps(session_update))
            print("\n2. Sent: session.update")

            # 3. Wait for session.updated
            response = await ws.recv()
            data = json.loads(response)
            print(f"3. Received: {data['type']}")

            # 4. Send audio (if test.wav exists)
            try:
                with wave.open("test.wav", "rb") as wav_file:
                    n_channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frame_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()

                    print(f"\n4. Audio file info:")
                    print(f"   Channels: {n_channels}")
                    print(f"   Sample width: {sample_width}")
                    print(f"   Frame rate: {frame_rate}")
                    print(f"   Frames: {n_frames}")

                    # Read audio data
                    audio_data = wav_file.readframes(n_frames)

                    # Convert to base64
                    audio_b64 = base64.b64encode(audio_data).decode("utf-8")

                    # Send in chunks
                    chunk_size = 32000  # 1 second at 16kHz, 16-bit
                    chunks = [
                        audio_b64[i : i + chunk_size] for i in range(0, len(audio_b64), chunk_size)
                    ]

                    print(f"\n5. Sending audio in {len(chunks)} chunks...")

                    for i, chunk in enumerate(chunks):
                        audio_msg = {
                            "event_id": f"audio_{i:03d}",
                            "type": "input_audio_buffer.append",
                            "audio": chunk,
                        }
                        await ws.send(json.dumps(audio_msg))
                        await asyncio.sleep(0.1)  # Simulate real-time streaming

                        # Try to receive any interim results
                        try:
                            response = await asyncio.wait_for(ws.recv(), timeout=0.01)
                            data = json.loads(response)
                            if data["type"] == "conversation.item.input_audio_transcription.text":
                                print(f"   Interim: {data.get('text', '')}{data.get('stash', '')}")
                        except asyncio.TimeoutError:
                            pass

                commit_msg = {"event_id": "event_commit", "type": "input_audio_buffer.commit"}
                await ws.send(json.dumps(commit_msg))
                print("\n6. Audio sent and committed...")

            except FileNotFoundError:
                print("\n4. test.wav not found, skipping audio test")
                print("   To test with audio, place a test.wav file in the current directory")

            finish_msg = {"event_id": "event_finish", "type": "session.finish"}
            await ws.send(json.dumps(finish_msg))
            print("   Sent: session.finish")

            # 6. Wait for final results
            print("\n7. Waiting for final results...")
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(response)
                    print(f"   Received: {data['type']}")

                    if data["type"] == "conversation.item.input_audio_transcription.completed":
                        print(f"   Final transcript: {data.get('transcript', '')}")
                    elif data["type"] == "session.finished":
                        print("\n8. Session finished!")
                        break
            except asyncio.TimeoutError:
                print("\n   Timeout waiting for final results")

    except (OSError, websockets.exceptions.InvalidStatus) as e:
        print(f"Error: Could not connect to {uri}: {e}")
        print("Make sure the server is running:")
        print("  ./scripts/start.sh")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3-ASR-Realtime Server Test")
    print("=" * 60)

    result = asyncio.run(test_websocket_server())
    sys.exit(result)
