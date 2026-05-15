from vieneu import Vieneu

REFERENCE_WAV = "reference_voice.wav"
REFERENCE_TEXT = (
    "Nhập chính xác nội dung mà người nói đã nói trong file reference_voice.wav."
)

TEXT = "Xin chào, đây là giọng nói tiếng Việt được clone bằng VieNeu-TTS Standard trên GPU."

OUTPUT_WAV = "output_standard_gpu.wav"


def create_voice_VI(reference_wav, reference_text, text, output_wav):
    # Standard mode + PyTorch GPU
    tts = Vieneu(
        mode="standard",
        backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B",
        backbone_device="cuda",
        # Codec mặc định là DistillNeuCodec.
        # Nếu muốn thử chất lượng cao hơn, có thể dùng NeuCodec full:
        # codec_repo="neuphonic/neucodec",
        # codec_device="cuda",
    )

    try:
        audio = tts.infer(
            text=text,
            ref_audio=reference_wav,
            ref_text=reference_text,
        )

        tts.save(audio, output_wav)
        print(f"Saved: {output_wav}")

    finally:
        tts.close()


def run_cmd(cmd):
    """Run shell command and raise readable error if failed."""
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{result.stderr}"
        )


def convert_mp3_to_wav(input_mp3: str, output_wav: str):
    """
    Convert MP3 reference voice to clean mono WAV.
    XTTS accepts speaker_wav; WAV reference is safest.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_mp3,
        "-ac",
        "1",
        "-ar",
        "22050",
        "-vn",
        output_wav,
    ]
    run_cmd(cmd)


def split_text(text: str, max_chars: int = 220):
    """
    Split long English text into smaller chunks.
    XTTS works better with short/medium chunks.
    """
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""

    for sent in sentences:
        if not sent:
            continue

        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return chunks


def concat_wavs(wav_files, output_wav):
    """
    Concatenate generated WAV files.
    """
    audio_all = []
    sample_rate = None

    for wav in wav_files:
        audio, sr = sf.read(wav)

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError("Sample rates do not match between generated chunks.")

        audio_all.append(audio)

    import numpy as np

    final_audio = np.concatenate(audio_all)
    sf.write(output_wav, final_audio, sample_rate)


def wav_to_mp3(input_wav: str, output_mp3: str):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_wav,
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "2",
        output_mp3,
    ]
    run_cmd(cmd)


def clone_voice_english(
    model_name, reference_mp3: str, text: str, output_path: str, use_gpu: bool = True
):
    reference_mp3 = str(Path(reference_mp3).resolve())
    output_path = str(Path(output_path).resolve())

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        reference_wav = tmpdir / "reference_voice.wav"

        print("[1/4] Converting MP3 reference voice to WAV...")
        convert_mp3_to_wav(reference_mp3, str(reference_wav))

        print("[2/4] Loading XTTS-v2 model...")
        tts = TTS(model_name, gpu=use_gpu)

        chunks = split_text(text)
        generated_wavs = []

        print(f"[3/4] Generating speech in English from {len(chunks)} text chunk(s)...")

        for i, chunk in enumerate(chunks, start=1):
            chunk_wav = tmpdir / f"chunk_{i:03d}.wav"
            print(f"  - Chunk {i}/{len(chunks)}: {chunk[:70]}...")

            tts.tts_to_file(
                text=chunk,
                speaker_wav=str(reference_wav),
                language="en",
                file_path=str(chunk_wav),
            )

            generated_wavs.append(str(chunk_wav))

        output_ext = Path(output_path).suffix.lower()

        if output_ext == ".wav":
            print("[4/4] Concatenating WAV chunks...")
            concat_wavs(generated_wavs, output_path)

        elif output_ext == ".mp3":
            final_wav = tmpdir / "final.wav"
            print("[4/4] Concatenating WAV chunks and converting to MP3...")
            concat_wavs(generated_wavs, str(final_wav))
            wav_to_mp3(str(final_wav), output_path)
        else:
            raise ValueError("Output path must end with .wav or .mp3")

    print(f"Done. Output saved to: {output_path}")


def create_voice_VI(reference_wav, reference_text, text, output_wav):
    tts = Vieneu(
        mode="standard",
        # PyTorch model repo
        backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B",
        backbone_device="cuda",
        # Quan trọng: tắt GGUF để ép dùng PyTorch/safetensors
        gguf_filename=None,
        # Cho codec chạy GPU luôn
        codec_repo="neuphonic/distill-neucodec",
        codec_device="cuda",
    )

    try:
        audio = tts.infer(
            text=text,
            ref_audio=reference_wav,
            ref_text=reference_text,
        )

        tts.save(audio, output_wav)
        print(f"Saved: {output_wav}")

    finally:
        tts.close()
