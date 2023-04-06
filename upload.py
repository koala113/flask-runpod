import whisper
import numpy as np
from flask import * 
import os
from stable_whisper import modify_model
import json
import stable_whisper
# from stable_whisper import stabilize_timestamps
app = Flask(__name__)  
# curr_dir = os.path.dirname(os.getcwd())
# model = whisper.load_model(curr_dir + "/pytorch_model.bin") 
with open('../pytorch_model.bin', 'rb') as f1:
    data = np.fromfile(f1, dtype=np.float32)
model = whisper.load_model('large')
modify_model(model)
# model.load_from_bytes(data.tobytes())

@app.route('/')  
def upload():  
    return render_template("index.html")  

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file'] 
        f.save(f.filename) 
        # result = model.transcribe(f.filename, language='fr', suppress_silence=True, ts_num=16)
        result = model.transcribe(f.filename, language='fr')# suppress_silence=True, ts_num=16)
        # print(result)
        result.to_srt_vtt('audio.srt', word_level=False)
        result.to_srt_vtt('output.srt', segment_level=False)
        result.to_ass('output.ass')
        result.save_as_json('audio.json')
        result = result.to_dict()
        stab_segments = result['segments']
        stab_segments = stab_segments.to_dict()
        sub_text = stab_segments['text']
        # first_segment_word_timestamps = stab_segments[0]['whole_word_timestamps']
        # stab_segments = stabilize_timestamps(result, top_focus=True)
        # print(stab_segments)
        # print(stab_segments[:]['words'])
        # print(stab_segments)
        # audio = whisper.load_audio(f.filename)
        # audio = whisper.pad_or_trim(audio) 
        # mel = whisper.log_mel_spectrogram(audio).to(model.device)
        # _, probs = model.detect_language(mel)
        # options = whisper.DecodingOptions()
        # result = whisper.decode(model, mel, options)
        # f.save(f.filename)  
        return render_template("index.html", name = result['text'], timestam = sub_text)  

if __name__ == '__main__':  
    app.run(host= '0.0.0.0', port='3000', debug = True)  
