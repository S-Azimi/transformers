from faster_whisper import WhisperModel

# انتخاب مدل و دستگاه
model = WhisperModel(
    "large-v3",
    device="cuda",          # یا "cpu" اگر GPU ندارید
    compute_type="float16"  # برای CPU از "int8" استفاده کنید
)

segments, info = model.transcribe(
    "mp3/Mohammad non banking.m4a",
    language="fa",          # مشخص کردن زبان فارسی (مهم!)
    beam_size=5,
    vad_filter=True         # حذف سکوت‌ها با VAD
)

print(f"زبان تشخیص داده شده: {info.language} ({info.language_probability:.2f})")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

