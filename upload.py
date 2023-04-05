import whisper
from flask import * 
import os
app = Flask(__name__)  
curr_dir = os.getcwd()
#model = whisper.load_model("../pytorch_model.bin") 

@app.route('/')  
def upload():  
    return render_template("index.html")  

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file'] 
        # audio = whisper.load_audio(f)
        # audio = whisper.pad_or_trim(audio) 
        # mel = whisper.log_mel_spectrogram(audio).to(model.device)
        # options = whisper.DecodingOptions()
        # result = whisper.decode(model, mel, options)
        #f.save(f.filename)  
        return render_template("index.html", name = curr_dir)  

if __name__ == '__main__':  
    app.run(host= '0.0.0.0', debug = True)  
