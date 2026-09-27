import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL = "/srv/ai/models/speech/tts/Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUT = "/srv/ai/workspace/media/speech-smoke-001/customvoice.wav"

text = (
    "真正限制本地人工智能模型使用体验的，"
    "往往并不只是模型本身的参数规模。"
)

print("Loading model...")

model = Qwen3TTSModel.from_pretrained(
    MODEL,
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

print("Supported speakers:", model.get_supported_speakers())
print("Supported languages:", model.get_supported_languages())

torch.cuda.synchronize()
t0 = time.time()

wavs, sr = model.generate_custom_voice(
    text=text,
    language="Chinese",
    speaker="Uncle_Fu",
    instruct="自然、清晰、克制的中文知识类视频旁白，语速中等，不要夸张。",
)

torch.cuda.synchronize()
elapsed = time.time() - t0

sf.write(OUT, wavs[0], sr)

print(f"Output: {OUT}")
print(f"Sample rate: {sr}")
print(f"Generation time: {elapsed:.2f}s")
