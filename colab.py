from datasets import load_dataset
import torchaudio

from transformers import pipeline

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pipe = pipeline("automatic-speech-recognition", model="bofenghuang/whisper-large-v2-french")
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="fr", task="transcribe")
# Load your own audio file
your_audio_file = "Audio123.wav"
waveform, sample_rate = torchaudio.load(your_audio_file)
required_sample_rate = 16000  
if sample_rate != required_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=required_sample_rate)
    waveform = resampler(waveform)
# Convert the input waveform to a NumPy array
waveform_np = waveform.squeeze().numpy()
# Run
generated_sentences = pipe(waveform_np, max_new_tokens=1000)["text"]
print(generated_sentences)