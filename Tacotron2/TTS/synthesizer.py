import io
import os

import numpy as np
import torch
import sys

from TTS.utils import AudioProcessor
from TTS.utils import load_config, setup_model, resource_path
from TTS.utils import load_speaker_mapping
from TTS.utils import text_to_seqvec, compute_style_mel, run_model, \
                        parse_outputs, trim_silence, inv_spectrogram, \
                        id_to_torch, synthesis
from TTS.utils import phonemes, symbols

import re
alphabets = r"([A-Za-z])"
prefixes = r"(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = r"(Inc|Ltd|Jr|Sr|Co)"
starters = r"(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = r"[.](com|net|org|io|gov)"


class Synthesizer(object):
    def __init__(self, config):
        self.wavernn = None
        self.config = config
        self.use_cuda = self.config.use_cuda
        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."

        tts_config_path = resource_path(self.config.tts_config)
        tts_checkpoint_path = resource_path(self.config.tts_checkpoint)
        self.load_tts(tts_checkpoint_path, tts_config_path, self.config.use_cuda)

        if self.config.wavernn_lib_path:
            wavernn_lib_path = resource_path(self.config.wavernn_lib_path)
            wavernn_path = resource_path(self.config.wavernn_path)
            self.load_wavernn(wavernn_lib_path, wavernn_path, wavernn_file, wavernn_config, self.config.use_cuda)

    def load_tts(self, tts_checkpoint, tts_config, use_cuda):
        print("\t[-] Loading TTS model ...")
        print("\t | > model config: ", tts_config)
        print("\t | > checkpoint file: ", tts_checkpoint)

        self.tts_config = load_config(tts_config)
        self.use_phonemes = self.tts_config.use_phonemes
        self.ap = AudioProcessor(**self.tts_config.audio)

        if self.use_phonemes:
            self.input_size = len(phonemes)
        else:
            self.input_size = len(symbols)

        if self.config.tts_speakers is not None:
            self.tts_speakers = load_speaker_mapping(os.path.join(model_path, self.config.tts_speakers))
            num_speakers = len(self.tts_speakers)
        else:
            num_speakers = 0

        self.tts_model = setup_model(self.input_size, num_speakers=num_speakers, c=self.tts_config)
        cp = torch.load(tts_checkpoint, map_location=torch.device('cpu'))
        self.tts_model.load_state_dict(cp['model'])
        if use_cuda:
            self.tts_model.cuda()
        self.tts_model.eval()
        self.tts_model.decoder.max_decoder_steps = 3000
        if 'r' in cp:
            self.tts_model.decoder.set_r(cp['r'])

    def load_wavernn(self, lib_path, model_path, model_file, model_config, use_cuda):
        sys.path.append(lib_path)
        from WaveRNN.models.wavernn import Model
        wavernn_config = os.path.join(model_path, model_config)
        model_file = os.path.join(model_path, model_file)

        print("\t[-] Loading WaveRNN model ...")
        print("\t | > model config: ", wavernn_config)
        print("\t | > model file: ", model_file)

        self.wavernn_config = load_config(wavernn_config)
        self.wavernn = Model(
            rnn_dims=512,
            fc_dims=512,
            mode=self.wavernn_config.mode,
            mulaw=self.wavernn_config.mulaw,
            pad=self.wavernn_config.pad,
            use_aux_net=self.wavernn_config.use_aux_net,
            use_upsample_net = self.wavernn_config.use_upsample_net,
            upsample_factors=self.wavernn_config.upsample_factors,
            feat_dims=80,
            compute_dims=128,
            res_out_dims=128,
            res_blocks=10,
            hop_length=self.ap.hop_length,
            sample_rate=self.ap.sample_rate,
        ).cuda()

        check = torch.load(model_file)
        self.wavernn.load_state_dict(check['model'])
        if use_cuda:
            self.wavernn.cuda()
        self.wavernn.eval()

    def save_wav(self, wav, path):
        wav = np.array(wav)
        self.ap.save_wav(wav, path)

    def split_into_sentences(self, text):
        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = re.sub(prefixes, "\\1<prd>", text)
        text = re.sub(websites, "<prd>\\1", text)
        if "Ph.D" in text:
            text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        text = re.sub(r"\s" + alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(acronyms+" "+starters, "\\1<stop> \\2", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" "+suffixes+"[.] "+starters, " \\1<stop> \\2", text)
        text = re.sub(" "+suffixes+"[.]", " \\1<prd>", text)
        text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
        if "”" in text:
            text = text.replace(".”", "”.")
        if "\"" in text:
            text = text.replace(".\"", "\".")
        if "!" in text:
            text = text.replace("!\"", "\"!")
        if "?" in text:
            text = text.replace("?\"", "\"?")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences

    def tts(self, text):
        text = f"{text}." # Adding a fullstop to the end
        wavs = []
        sens = self.split_into_sentences(text)
        if not sens:
            sens = [text+'.']
        for sen in sens:
            inputs = text_to_seqvec(sen, self.tts_config, self.use_cuda)
            
            decoder_output, postnet_output, alignments, _ = run_model(
                self.tts_model, inputs, self.tts_config, False, None, None)
            
            postnet_output, decoder_output, _ = parse_outputs(postnet_output, decoder_output, alignments)

            if self.wavernn:
                postnet_output = postnet_output[0].data.cpu().numpy()
                wav = self.wavernn.generate(
                    torch.FloatTensor(postnet_output.T).unsqueeze(0).cuda(), 
                    batched=self.config.is_wavernn_batched, 
                    target=11000, 
                    overlap=550)
            else:
                wav = inv_spectrogram(
                    postnet_output, 
                    self.ap, 
                    self.tts_config)

            wav = trim_silence(wav, self.ap)
            wavs += list(wav)
            wavs += [0] * 10000

        out = io.BytesIO()
        self.save_wav(wavs, out)
        return out
