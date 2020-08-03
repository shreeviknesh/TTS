import io
import os
import torch
import numpy as np

from .utils import text_to_sequence, phoneme_to_sequence
from .utils import symbols, phonemes
from .utils import load_config, AudioProcessor

from .models import Tacotron

class Synthesizer(object):
    def load_model(self, model_path, model_name, model_config, use_cuda):
        self.model_file = os.path.join(model_path, model_name)
        model_config = os.path.join(model_path, model_config)

        print("\t[-] Loading model...")
        print("\t | > model config: ", model_config)
        print("\t | > model file: ", self.model_file)

        config = load_config(model_config)
        self.config = config
        self.use_cuda = use_cuda
        self.ap = AudioProcessor(**config.audio)

        num_chars = len(phonemes) if config.use_phonemes else len(symbols)

        self.model = Tacotron(
            num_chars,
            config.embedding_size,
            self.ap.num_freq,
            self.ap.num_mels,
            config.r,
            config.memory_size
        )

        # load model state
        if use_cuda:
            cp = torch.load(self.model_file)
        else:
            cp = torch.load(self.model_file, map_location=lambda storage, loc: storage)

        # load the model
        self.model.load_state_dict(cp['model'])
        if use_cuda:
            self.model.cuda()

        self.model.eval()

    def save_wav(self, wav, path):
        wav = np.array(wav)
        self.ap.save_wav(wav, path)

    def tts(self, text):
        text_cleaner = [self.config.text_cleaner]
        wavs = []
        for sen in text.split('.'):
            if len(sen) < 3:
                continue
            sen = sen.strip()
            sen += '.'
            sen = sen.strip()
            if self.config.use_phonemes:
                seq = np.asarray(
                    phoneme_to_sequence(sen, text_cleaner, self.config.phoneme_language),
                    dtype=np.int32)
            else:
                seq = np.asarray(text_to_sequence(sen, text_cleaner), dtype=np.int32)
            chars_var = torch.from_numpy(seq).unsqueeze(0).long()
            if self.use_cuda:
                chars_var = chars_var.cuda()
            mel_out, linear_out, alignments, stop_tokens = self.model.forward(
                chars_var)
            linear_out = linear_out[0].data.cpu().numpy()
            wav = self.ap.inv_spectrogram(linear_out.T)
            out = io.BytesIO()
            wavs += list(wav)
            wavs += [0] * 10000
        self.save_wav(wavs, out)
        return out
