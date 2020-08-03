import os, sys
import subprocess
from flask import Flask, Response, request, render_template, send_file

# Helper functions
def execute_command(cmd, quit_on_error=False):
    retval = subprocess.call(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if retval != 0 and quit_on_error:
        print(f"\t\t Some error occured while executing `{cmd}`")
        sys.exit()
    return retval

def resource_path(relative_path):
    bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(bundle_dir, relative_path)

# Getting dependencies ready
print("[+] Installing required Ubuntu packages.")
packages = ["espeak", "libsndfile1"]
is_updated = False

for item in packages:
    retval = execute_command(f"dpkg -s {item}")
    if retval != 0:
        if is_updated == False:
            print(f"\t[-] Updating apt-get.")
            retval = execute_command("sudo apt-get -y update", quit_on_error=True)

        print(f"\t[-] Installing {item}.")
        retval = execute_command(f"sudo apt-get -y install {item}", quit_on_error=True)

    else:
        print(f"\t[-] {item} already installed.")

# Executing the service
print("[+] The service is now starting.")
from TTS.synthesizer import Synthesizer

model_path = resource_path('models')
model_name = 'best_model.pth.tar'
model_config = 'config.json'
use_cuda = False

app = Flask(__name__, template_folder=resource_path('templates'))
synthesizer = Synthesizer()
synthesizer.load_model(model_path, model_name, model_config, use_cuda)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tts', methods=['GET'])
def tts():
    text = request.args.get('text')
    print(" > Model input: {}".format(text))
    data = synthesizer.tts(text)
    return send_file(data, mimetype='audio/wav')

app.run(debug=False, host='0.0.0.0', port=5002)
