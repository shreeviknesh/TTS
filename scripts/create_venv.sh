#!/bin/sh

pip3 install virtualenv
python3 -m virtualenv -p python3 .
. bin/activate
pip install numpy==1.19.0
pip install torch==1.5.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install librosa==0.7.2
pip install Unidecode==1.1.1
pip install phonemizer==2.2
pip install flask==1.1.2
pip install numba==0.48
pip install pyinstaller
