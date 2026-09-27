import json
import time
import torch
from qwen_asr import Qwen3ForcedAligner

MODEL = "/srv/ai/models/speech/asr/Qwen3-ForcedAligner-0.6B"
AUDIO = "/srv/ai/workspace/media/speech-smoke-001/customvoice.wav"
OUT = "/srv/ai/workspace/media/speech-smoke-001/alignment.json"

TEXT = (
    "真正限制本地人工智能模型使用体验的，"
    "往往并不只是模型本身的参数规模。"
)

print("Loading forced aligner...")

model = Qwen3ForcedAligner.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

torch.cuda.synchronize()
t0 = time.time()

results = model.align(
    audio=AUDIO,
    text=TEXT,
    language="Chinese",
)

torch.cuda.synchronize()
elapsed = time.time() - t0

items = [
    {
        "text": item.text,
        "start": item.start_time,
        "end": item.end_time,
    }
    for item in results[0]
]

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print(json.dumps(items, ensure_ascii=False, indent=2))
print()
print(f"Items: {len(items)}")
print(f"Alignment time: {elapsed:.2f}s")
print(f"Output: {OUT}")
