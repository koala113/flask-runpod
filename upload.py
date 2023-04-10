import whisper
import numpy as np
from flask import * 
import os
from stable_whisper import modify_model
import json
import stable_whisper
import re
import torch


# from stable_whisper import stabilize_timestamps
app = Flask(__name__)  
# curr_dir = os.path.dirname(os.getcwd())
# model = whisper.load_model(curr_dir + "/pytorch_model.bin") 
def hf_to_whisper_states(text):
    text = re.sub('.layers.', '.blocks.', text)
    text = re.sub('.self_attn.', '.attn.', text)
    text = re.sub('.q_proj.', '.query.', text)
    text = re.sub('.k_proj.', '.key.', text)
    text = re.sub('.v_proj.', '.value.', text)
    text = re.sub('.out_proj.', '.out.', text)
    text = re.sub('.fc1.', '.mlp.0.', text)
    text = re.sub('.fc2.', '.mlp.2.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.encoder_attn.', '.cross_attn.', text)
    text = re.sub('.cross_attn.ln.', '.cross_attn_ln.', text)
    text = re.sub('.embed_positions.weight', '.positional_embedding', text)
    text = re.sub('.embed_tokens.', '.token_embedding.', text)
    text = re.sub('model.', '', text)
    text = re.sub('attn.layer_norm.', 'attn_ln.', text)
    text = re.sub('.final_layer_norm.', '.mlp_ln.', text)
    text = re.sub('encoder.layer_norm.', 'encoder.ln_post.', text)
    text = re.sub('decoder.layer_norm.', 'decoder.ln.', text)
    return text
hf_state_dict = torch.load('../pytorch_model.bin') 
for key in list(hf_state_dict.keys())[:]:
    new_key = hf_to_whisper_states(key)
    hf_state_dict[new_key] = hf_state_dict.pop(key)
model = whisper.load_model('large')
model.load_state_dict(hf_state_dict)
# with open('../pytorch_model.bin', 'rb') as f1:
#     data = np.fromfile(f1, dtype=np.float32)
# model = whisper.load_model('large')
# modify_model(model)
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
        # result.to_srt_vtt('audio.srt', word_level=False)
        # result.to_srt_vtt('output.srt', segment_level=False)
        # result.to_ass('output.ass')
        # result.save_as_json('audio.json')
        # result = result.to_dict()
        stab_segments = result['segments']
        count_sentence = len(stab_segments)
        print_list = []
        for i in range(0, count_sentence):
            start = stab_segments[i]['start']
            end = stab_segments[i]['end']
            text = stab_segments[i]['text']
            total = str(start) + '~' + str(end) + ':' + str(text)
            print_list.append(total)
            # print_list.append('\n')
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
        data = {
            'result' : result['text'],
            'timestam' : print_list
        }
        json_data = json.dumps(data)
        response = Response(json_data, content_type = 'application/json')
        return response
        # return render_template("index.html", name = result['text'], timestam = print_list)  

if __name__ == '__main__':  
    app.run(host= '0.0.0.0', port='3000', debug = True)  
