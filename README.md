# Qwen3-ASR-Realtime Server

兼容阿里云 Qwen3-ASR-realtime API 的私有部署 WebSocket 服务，基于 vLLM 0.14.0。

## 快速开始

### Docker 部署 (推荐)

```bash
docker run -d \
  --gpus all --ipc host \
  -p 28787:28787 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e QWEN3_ASR_MODEL_PATH=Qwen/Qwen3-ASR-0.6B \
  rookiezoe/qwen3-asr-realtime:latest
```

### 本地开发

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 初始化
cp .env.example .env
uv sync

# 启动
uv run python main.py
```

## 配置

编辑 `.env`:

```bash
QWEN3_ASR_MODEL_PATH=Qwen/Qwen3-ASR-0.6B  # HF ID / 绝对路径
GPU_MEMORY_UTILIZATION=0.8
MODEL_DTYPE=auto  # auto/half/float16/bfloat16
```

完整配置见 [.env.example](.env.example)

## API

### WebSocket

```
ws://localhost:28787/api-ws/v1/realtime
```

### HTTP

| 端点 | 说明 |
|------|------|
| `/health` | 健康检查 |
| `/docs` | API 文档 (含 WebSocket 协议说明) |

## 使用示例

```python
import asyncio, base64, json, websockets

async def recognize():
    async with websockets.connect("ws://localhost:28787/api-ws/v1/realtime") as ws:
        await ws.recv()  # session.created
        
        # 配置 (Manual 模式)
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"turn_detection": None}
        }))
        
        # 发送音频
        with open("audio.wav", "rb") as f:
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(f.read()).decode()
            }))
        
        # 提交并获取结果
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        
        async for msg in ws:
            data = json.loads(msg)
            if data["type"].endswith(".completed"):
                print(data["transcript"])
                break

asyncio.run(recognize())
```

VAD 模式: 设置 `turn_detection: {"type": "server_vad"}` 自动检测语音边界。

## 项目结构

```
├── docker/          # Docker 相关文件
├── src/             # 源代码
├── demo/            # Web/SDK 演示
├── tests/           # 测试
└── main.py          # 入口
```

## License

MIT - 详见 [LICENSE](LICENSE)

## 致谢

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (Apache 2.0)
- [vLLM](https://github.com/vllm-project/vllm)
