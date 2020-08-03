## Mozilla TTS

### Contents
1. Tacotron
2. Tacotron2

Both directories contain different programs to run Tacotron and Tacotron2 models respectively. They contain the model checkpoint file under the `models` directory.

### Instructions to run the program:
1. Run `source scripts/create_venv.sh`. This will create a virtualenv, activate it and install all the requirements.
2. Run either `python Tacotron/Tacotron.py` or `python Tacotron2/Tacotron2.py` depending on which model is needed.
3. Go to **localhost:5002** to test the TTS.

### Instructions to build to an executable:
1. Run `source scripts/create_venv.sh`. This will create a virtualenv, activate it and install all the requirements.
2. Create the executable using `source scripts/bundle_tacotron.sh` or `source scripts\bundle_tacotron2.sh` depending on the model. Now a directory called _dist_ will be created in the directory.
3. Run this executable using `./dist/Tacotron/Tacotron` or `./dist/Tacotron2/Tacotron2`
4. Zip this dist directory directory using `zip Tacotron.zip dist/Tacotron` or `zip Tacotron2.zip dist/Tacotron2`