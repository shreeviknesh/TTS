import os
import sys
import json
from flask import Flask, request, render_template, send_file
import subprocess

# Using _MEIPASS if it's a bundle or the root dir if it's not
os.environ['tts_base_dir'] = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))

# Helper functions
def execute_command(cmd, quit_on_error=False):
    retval = subprocess.call(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if retval != 0 and quit_on_error:
        print(f"\t\t Some error occured while executing `{cmd}`")
        sys.exit()
    return retval

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
from TTS.utils import Map, resource_path
from TTS.synthesizer import Synthesizer

with open(resource_path('conf.json'), 'rt') as file:
    args = Map(json.load(file))

synthesizer = Synthesizer(args)
app = Flask(__name__, template_folder=resource_path('templates'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tts', methods=['GET'])
def tts():
    text = request.args.get('text')
    print(" > Model input: {}".format(text))
    data = synthesizer.tts(text)
    return send_file(data, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(debug=args.debug, host='0.0.0.0', port=args.port)
