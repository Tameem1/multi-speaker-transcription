import os
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
import sys

def transcribe_speaker_diarization(audio_file: str, hf_token: str) -> str:
    """
    Transcribe an English audio file with multiple speakers,
    labeling each segment by speaker ID and time range.

    :param audio_file: Path to the audio file (e.g. 'conversation.wav')
    :param hf_token: Hugging Face authentication token (for pyannote.audio)
    :return: Combined transcription string with timestamps and speaker labels
    """
    # -------------------------------------------------------------------------
    # 0. Convert M4A to WAV if needed
    # -------------------------------------------------------------------------
    if audio_file.lower().endswith(".m4a"):
        audio = AudioSegment.from_file(audio_file, format="m4a")
        wav_file = audio_file.replace(".m4a", ".wav")
        audio.export(wav_file, format="wav")
        audio_file = wav_file  # Use the WAV file for processing
    else:
        wav_file = None
    # -------------------------------------------------------------------------
    # 1. Initialize speaker diarization pipeline
    #    (Requires valid HF token to access the pyannote/speaker-diarization model)
    # -------------------------------------------------------------------------
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token=hf_token)

    # -------------------------------------------------------------------------
    # 2. Perform diarization to get speaker segments
    # -------------------------------------------------------------------------
    diarization_result = pipeline(audio_file)

    # -------------------------------------------------------------------------
    # 3. Load Whisper "large-v2" model on GPU if available
    # -------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model("large-v2", device=device)

    # -------------------------------------------------------------------------
    # 4. Use pydub to load the entire audio file
    # -------------------------------------------------------------------------
    audio = AudioSegment.from_file(audio_file)

    # -------------------------------------------------------------------------
    # 5. Iterate through speaker segments, slice audio, transcribe
    # -------------------------------------------------------------------------
    segments = []

    # diarization_result is a pyannote.core.annotation.Annotation
    # we iterate with "itertracks(yield_label=True)" to get segment and speaker
    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        start_time = segment.start
        end_time = segment.end

        # Slice audio with pydub (works in milliseconds)
        audio_chunk = audio[start_time * 1000 : end_time * 1000]

        # Create a temporary WAV file for this chunk
        temp_filename = f"temp_{speaker}_{start_time:.2f}.wav"
        audio_chunk.export(temp_filename, format="wav")

        # Transcribe the chunk using Whisper
        # Language hint set to English for better performance
        result = whisper_model.transcribe(temp_filename, language="en")

        segments.append({
            "speaker": speaker,
            "start": start_time,
            "end": end_time,
            "text": result["text"].strip()
        })

        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    # -------------------------------------------------------------------------
    # 6. Sort segments by start time (in case they are out of order)
    # -------------------------------------------------------------------------
    segments.sort(key=lambda x: x["start"])

    # -------------------------------------------------------------------------
    # 7. Construct a readable transcript
    # -------------------------------------------------------------------------
    transcript_str = ""
    for seg in segments:
        transcript_str += (
            f"Speaker {seg['speaker']} "
            f"({seg['start']:.2f}s - {seg['end']:.2f}s): {seg['text']}\n"
        )
    # Clean up temporary WAV file if created
    if wav_file and os.path.exists(wav_file):
        os.remove(wav_file)
    return transcript_str


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python transcribe_with_diarization.py <audio_file> <hf_token>")
        sys.exit(1)

    AUDIO_FILE = sys.argv[1]
    HF_TOKEN = sys.argv[2]
    
    transcript = transcribe_speaker_diarization(AUDIO_FILE, HF_TOKEN)
    print(transcript)