import os
import sys
import subprocess

import re
import glob
import random
import shutil
import datetime
import json
import collections

import torch
import librosa
import soundfile as sf
import numpy as np
import scipy.io
import scipy.signal
import phonemizer
from phonemizer.phonemize import phonemize
from unidecode import unidecode

from TTS.models import Tacotron2

# datasets.preprocess.py
def load_meta_data(datasets):
    meta_data_train_all = []
    meta_data_eval_all = []
    for dataset in datasets:
        name = dataset['name']
        root_path = dataset['path']
        meta_file_train = dataset['meta_file_train']
        meta_file_val = dataset['meta_file_val']
        preprocessor = get_preprocessor_by_name(name)

        meta_data_train = preprocessor(root_path, meta_file_train)
        if meta_file_val is None:
            meta_data_eval, meta_data_train = split_dataset(meta_data_train)
        else:
            meta_data_eval = preprocessor(root_path, meta_file_val)
        meta_data_train_all += meta_data_train
        meta_data_eval_all += meta_data_eval
    return meta_data_train_all, meta_data_eval_all

def get_preprocessor_by_name(name):
    """Returns the respective preprocessing function."""
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())

def tweb(root_path, meta_file):
    """Normalize TWEB dataset.
    https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "tweb"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('\t')
            wav_file = os.path.join(root_path, cols[0] + '.wav')
            text = cols[1]
            items.append([text, wav_file, speaker_name])
    return items

def mozilla_old(root_path, meta_file):
    """Normalizes Mozilla meta data files to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mozilla_old"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            batch_no = int(cols[1].strip().split("_")[0])
            wav_folder = "batch{}".format(batch_no)
            wav_file = os.path.join(root_path, wav_folder, "wavs_no_processing", cols[1].strip())
            text = cols[0].strip()
            items.append([text, wav_file, speaker_name])
    return items

def mozilla(root_path, meta_file):
    """Normalizes Mozilla meta data files to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mozilla"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = cols[1].strip()
            text = cols[0].strip()
            wav_file = os.path.join(root_path, "wavs", wav_file)
            items.append([text, wav_file, speaker_name])
    return items

def mailabs(root_path, meta_files=None):
    """Normalizes M-AI-Labs meta data files to TTS format"""
    speaker_regex = re.compile("by_book/(male|female)/(?P<speaker_name>[^/]+)/")
    if meta_files is None:
        csv_files = glob(root_path+"/**/metadata.csv", recursive=True)
    else:
        csv_files = meta_files
    # meta_files = [f.strip() for f in meta_files.split(",")]
    items = []
    for csv_file in csv_files:
        txt_file = os.path.join(root_path, csv_file)
        folder = os.path.dirname(txt_file)
        # determine speaker based on folder structure...
        speaker_name_match = speaker_regex.search(txt_file)
        if speaker_name_match is None:
            continue
        speaker_name = speaker_name_match.group("speaker_name")
        print(" | > {}".format(csv_file))
        with open(txt_file, 'r') as ttf:
            for line in ttf:
                cols = line.split('|')
                if meta_files is None:
                    wav_file = os.path.join(folder, 'wavs', cols[0] + '.wav')
                else:
                    wav_file = os.path.join(root_path, folder.replace("metadata.csv", ""), 'wavs', cols[0] + '.wav')
                if os.path.isfile(wav_file):
                    text = cols[1].strip()
                    items.append([text, wav_file, speaker_name])
                else:
                    raise RuntimeError("> File %s is not exist!"%(wav_file))
    return items

def ljspeech(root_path, meta_file):
    """Normalizes the Nancy meta data file to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "ljspeech"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = os.path.join(root_path, 'wavs', cols[0] + '.wav')
            text = cols[1]
            items.append([text, wav_file, speaker_name])
    return items

def nancy(root_path, meta_file):
    """Normalizes the Nancy meta data file to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "nancy"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            utt_id = line.split()[1]
            text = line[line.find('"') + 1:line.rfind('"') - 1]
            wav_file = os.path.join(root_path, "wavn", utt_id + ".wav")
            items.append([text, wav_file, speaker_name])
    return items

def common_voice(root_path, meta_file):
    """Normalize the common voice meta data file to TTS format."""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            if line.startswith("client_id"):
                continue
            cols = line.split("\t")
            text = cols[2]
            speaker_name = cols[0]
            wav_file = os.path.join(root_path, "clips", cols[1] + ".wav")
            items.append([text, wav_file, speaker_name])
    return items

def libri_tts(root_path, meta_files=None):
    """https://ai.google/tools/datasets/libri-tts/"""
    items = []
    if meta_files is None:
        meta_files = glob(f"{root_path}/**/*trans.tsv", recursive=True)
    for meta_file in meta_files:
        _meta_file = os.path.basename(meta_file).split('.')[0]
        speaker_name = _meta_file.split('_')[0]
        chapter_id = _meta_file.split('_')[1]
        _root_path = os.path.join(root_path, f"{speaker_name}/{chapter_id}")
        with open(meta_file, 'r') as ttf:
            for line in ttf:
                cols = line.split('\t')
                wav_file = os.path.join(_root_path, cols[0] + '.wav')
                text = cols[1]
                items.append([text, wav_file, speaker_name])
    for item in items:
        assert os.path.exists(item[1]), f" [!] wav file is not exist - {item[1]}"
    return items

# datasets.TTSDataset.py
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 outputs_per_step,
                 text_cleaner,
                 ap,
                 meta_data,
                 batch_group_size=0,
                 min_seq_len=0,
                 max_seq_len=float("inf"),
                 use_phonemes=True,
                 phoneme_cache_path=None,
                 phoneme_language="en-us",
                 enable_eos_bos=False,
                 verbose=False):
        """
        Args:
            outputs_per_step (int): number of time frames predicted per step.
            text_cleaner (str): text cleaner used for the dataset.
            ap (TTS.utils.AudioProcessor): audio processor object.
            meta_data (list): list of dataset instances.
            batch_group_size (int): (0) range of batch randomization after sorting
                sequences by length.
            min_seq_len (int): (0) minimum sequence length to be processed
                by the loader.
            max_seq_len (int): (float("inf")) maximum sequence length.
            use_phonemes (bool): (true) if true, text converted to phonemes.
            phoneme_cache_path (str): path to cache phoneme features.
            phoneme_language (str): one the languages from
                https://github.com/bootphon/phonemizer#languages
            enable_eos_bos (bool): enable end of sentence and beginning of sentences characters.
            verbose (bool): print diagnostic information.
        """
        self.batch_group_size = batch_group_size
        self.items = meta_data
        self.outputs_per_step = outputs_per_step
        self.sample_rate = ap.sample_rate
        self.cleaners = text_cleaner
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.ap = ap
        self.use_phonemes = use_phonemes
        self.phoneme_cache_path = phoneme_cache_path
        self.phoneme_language = phoneme_language
        self.enable_eos_bos = enable_eos_bos
        self.verbose = verbose
        if use_phonemes and not os.path.isdir(phoneme_cache_path):
            os.makedirs(phoneme_cache_path, exist_ok=True)
        if self.verbose:
            print("\n > DataLoader initialization")
            print(" | > Use phonemes: {}".format(self.use_phonemes))
            if use_phonemes:
                print("   | > phoneme language: {}".format(phoneme_language))
            print(" | > Number of instances : {}".format(len(self.items)))
        self.sort_items()

    def load_wav(self, filename):
        audio = self.ap.load_wav(filename)
        return audio

    @staticmethod
    def load_np(filename):
        data = np.load(filename).astype('float32')
        return data

    def _generate_and_cache_phoneme_sequence(self, text, cache_path):
        """generate a phoneme sequence from text.

        since the usage is for subsequent caching, we never add bos and
        eos chars here. Instead we add those dynamically later; based on the
        config option."""
        phonemes = phoneme_to_sequence(text, [self.cleaners],
                                       language=self.phoneme_language,
                                       enable_eos_bos=False)
        phonemes = np.asarray(phonemes, dtype=np.int32)
        np.save(cache_path, phonemes)
        return phonemes

    def _load_or_generate_phoneme_sequence(self, wav_file, text):
        file_name = os.path.basename(wav_file).split('.')[0]
        cache_path = os.path.join(self.phoneme_cache_path,
                                  file_name + '_phoneme.npy')
        try:
            phonemes = np.load(cache_path)
        except FileNotFoundError:
            phonemes = self._generate_and_cache_phoneme_sequence(text,
                                                                 cache_path)
        except (ValueError, IOError):
            print(" > ERROR: failed loading phonemes for {}. "
                  "Recomputing.".format(wav_file))
            phonemes = self._generate_and_cache_phoneme_sequence(text,
                                                                 cache_path)
        if self.enable_eos_bos:
            phonemes = pad_with_eos_bos(phonemes)
            phonemes = np.asarray(phonemes, dtype=np.int32)
        return phonemes

    def load_data(self, idx):
        text, wav_file, speaker_name = self.items[idx]
        wav = np.asarray(self.load_wav(wav_file), dtype=np.float32)

        if self.use_phonemes:
            text = self._load_or_generate_phoneme_sequence(wav_file, text)
        else:
            text = np.asarray(
                text_to_sequence(text, [self.cleaners]), dtype=np.int32)

        assert text.size > 0, self.items[idx][1]
        assert wav.size > 0, self.items[idx][1]

        sample = {
            'text': text,
            'wav': wav,
            'item_idx': self.items[idx][1],
            'speaker_name': speaker_name
        }
        return sample

    def sort_items(self):
        r"""Sort instances based on text length in ascending order"""
        lengths = np.array([len(ins[0]) for ins in self.items])

        idxs = np.argsort(lengths)
        new_items = []
        ignored = []
        for i, idx in enumerate(idxs):
            length = lengths[idx]
            if length < self.min_seq_len or length > self.max_seq_len:
                ignored.append(idx)
            else:
                new_items.append(self.items[idx])
        # shuffle batch groups
        if self.batch_group_size > 0:
            for i in range(len(new_items) // self.batch_group_size):
                offset = i * self.batch_group_size
                end_offset = offset + self.batch_group_size
                temp_items = new_items[offset:end_offset]
                random.shuffle(temp_items)
                new_items[offset:end_offset] = temp_items
        self.items = new_items

        if self.verbose:
            print(" | > Max length sequence: {}".format(np.max(lengths)))
            print(" | > Min length sequence: {}".format(np.min(lengths)))
            print(" | > Avg length sequence: {}".format(np.mean(lengths)))
            print(" | > Num. instances discarded by max-min (max={}, min={}) seq limits: {}".format(
                self.max_seq_len, self.min_seq_len, len(ignored)))
            print(" | > Batch group size: {}.".format(self.batch_group_size))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.load_data(idx)

    def collate_fn(self, batch):
        r"""
            Perform preprocessing and create a final data batch:
            1. Sort batch instances by text-length
            2. Convert Audio signal to Spectrograms.
            3. PAD sequences wrt r.
            4. Load to Torch.
        """

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.Mapping):

            text_lenghts = np.array([len(d["text"]) for d in batch])

            # sort items with text input length for RNN efficiency
            text_lenghts, ids_sorted_decreasing = torch.sort(
                torch.LongTensor(text_lenghts), dim=0, descending=True)

            wav = [batch[idx]['wav'] for idx in ids_sorted_decreasing]
            item_idxs = [
                batch[idx]['item_idx'] for idx in ids_sorted_decreasing
            ]
            text = [batch[idx]['text'] for idx in ids_sorted_decreasing]
            speaker_name = [batch[idx]['speaker_name']
                            for idx in ids_sorted_decreasing]

            # compute features
            mel = [self.ap.melspectrogram(w).astype('float32') for w in wav]
            linear = [self.ap.spectrogram(w).astype('float32') for w in wav]

            mel_lengths = [m.shape[1] for m in mel]

            # compute 'stop token' targets
            stop_targets = [
                np.array([0.] * (mel_len - 1) + [1.]) for mel_len in mel_lengths
            ]

            # PAD stop targets
            stop_targets = prepare_stop_target(stop_targets,
                                               self.outputs_per_step)

            # PAD sequences with longest instance in the batch
            text = prepare_data(text).astype(np.int32)
            wav = prepare_data(wav)

            # PAD features with longest instance
            linear = prepare_tensor(linear, self.outputs_per_step)
            mel = prepare_tensor(mel, self.outputs_per_step)
            assert mel.shape[2] == linear.shape[2]

            # B x D x T --> B x T x D
            linear = linear.transpose(0, 2, 1)
            mel = mel.transpose(0, 2, 1)

            # convert things to pytorch
            text_lenghts = torch.LongTensor(text_lenghts)
            text = torch.LongTensor(text)
            linear = torch.FloatTensor(linear).contiguous()
            mel = torch.FloatTensor(mel).contiguous()
            mel_lengths = torch.LongTensor(mel_lengths)
            stop_targets = torch.FloatTensor(stop_targets)

            return text, text_lenghts, speaker_name, linear, mel, mel_lengths, \
                   stop_targets, item_idxs

        raise TypeError(("batch must contain tensors, numbers, dicts or lists;\
                         found {}".format(type(batch[0]))))

# utils.text.symbols.py
_pad = '_'
_eos = '~'
_bos = '^'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '
_punctuations = '!\'(),-.:;? '
_phoneme_punctuations = '.!;:,?'

# Phonemes definition
_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
_suprasegmentals = 'ˈˌːˑ'
_other_symbols = 'ʍwɥʜʢʡɕʑɺɧ'
_diacrilics = 'ɚ˞ɫ'
_phonemes = sorted(list(_vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics))

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in _phonemes]

# Export all symbols:
symbols = [_pad, _eos, _bos] + list(_characters) + _arpabet
phonemes = [_pad, _eos, _bos] + list(_phonemes) + list(_punctuations)

# utils.text.__init__.py
# Mappings from symbol to numeric ID and vice versa:
_SYMBOL_TO_ID = {s: i for i, s in enumerate(symbols)}
_ID_TO_SYMBOL = {i: s for i, s in enumerate(symbols)}

_PHONEMES_TO_ID = {s: i for i, s in enumerate(phonemes)}
_ID_TO_PHONEMES = {i: s for i, s in enumerate(phonemes)}

# Regular expression matching text enclosed in curly braces:
_CURLY_RE = re.compile(r'(.*?)\{(.+?)\}(.*)')
PHONEME_PUNCTUATION_PATTERN = r'['+_phoneme_punctuations+']+'

def text2phone(text, language):
    '''
    Convert graphemes to phonemes.
    '''
    seperator = phonemizer.separator.Separator(' |', '', '|')
    #try:
    punctuations = re.findall(PHONEME_PUNCTUATION_PATTERN, text)
    ph = phonemize(text, separator=seperator, strip=False, njobs=1, backend='espeak', language=language)
    ph = ph[:-1].strip() # skip the last empty character
    # Replace \n with matching punctuations.
    if punctuations:
        # if text ends with a punctuation.
        if text[-1] == punctuations[-1]:
            for punct in punctuations[:-1]:
                ph = ph.replace('| |\n', '|'+punct+'| |', 1)
            try:
                ph = ph + punctuations[-1]
            except:
                print(text)
        else:
            for punct in punctuations:
                ph = ph.replace('| |\n', '|'+punct+'| |', 1)
    return ph

def pad_with_eos_bos(phoneme_sequence):
    return [_PHONEMES_TO_ID[_bos]] + list(phoneme_sequence) + [_PHONEMES_TO_ID[_eos]]

def phoneme_to_sequence(text, cleaner_names, language, enable_eos_bos=False):
    sequence = []
    text = text.replace(":", "")
    clean_text = _clean_text(text, cleaner_names)
    to_phonemes = text2phone(clean_text, language)
    if to_phonemes is None:
        print("!! After phoneme conversion the result is None. -- {} ".format(clean_text))
    # iterate by skipping empty strings - NOTE: might be useful to keep it to have a better intonation.
    for phoneme in filter(None, to_phonemes.split('|')):
        sequence += _phoneme_to_sequence(phoneme)
    # Append EOS char
    if enable_eos_bos:
        sequence = pad_with_eos_bos(sequence)
    return sequence

def sequence_to_phoneme(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _ID_TO_PHONEMES:
            s = _ID_TO_PHONEMES[symbol_id]
            result += s
    return result.replace('}{', ' ')

def text_to_sequence(text, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

      The text can optionally have ARPAbet sequences enclosed in curly braces embedded
      in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

      Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through

      Returns:
        List of integers corresponding to the symbols in the text
    '''
    sequence = []
    # Check for curly braces and treat their contents as ARPAbet:
    while text:
        m = _CURLY_RE.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(
            _clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)
    return sequence

def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _ID_TO_SYMBOL:
            s = _ID_TO_SYMBOL[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')

def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = globals()[name]
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text

def _symbols_to_sequence(syms):
    return [_SYMBOL_TO_ID[s] for s in syms if _should_keep_symbol(s)]

def _phoneme_to_sequence(phons):
    return [_PHONEMES_TO_ID[s] for s in list(phons) if _should_keep_phoneme(s)]

def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])

def _should_keep_symbol(s):
    return s in _SYMBOL_TO_ID and s not in ['~', '^', '_']

def _should_keep_phoneme(p):
    return p in _PHONEMES_TO_ID and p not in ['~', '^', '_']

# utils.text.cleaners.py
_whitespace_re = re.compile(r'\s+')
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1])
                  for x in [
                      ('mrs', 'misess'),
                      ('mr', 'mister'),
                      ('dr', 'doctor'),
                      ('st', 'saint'),
                      ('co', 'company'),
                      ('jr', 'junior'),
                      ('maj', 'major'),
                      ('gen', 'general'),
                      ('drs', 'doctors'),
                      ('rev', 'reverend'),
                      ('lt', 'lieutenant'),
                      ('hon', 'honorable'),
                      ('sgt', 'sergeant'),
                      ('capt', 'captain'),
                      ('esq', 'esquire'),
                      ('ltd', 'limited'),
                      ('col', 'colonel'),
                      ('ft', 'fort'),
                  ]]

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def expand_numbers(text):
    return normalize_numbers(text)

def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text).strip()

def convert_to_ascii(text):
    return unidecode(text)

def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text

def phoneme_cleaners(text):
    '''Pipeline for phonemes mode, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text

# utils.text.number_norm.py
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'([0-9]+)(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')

_units = [
    '', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen'
]
_tens = [
    '',
    'ten',
    'twenty',
    'thirty',
    'forty',
    'fifty',
    'sixty',
    'seventy',
    'eighty',
    'ninety',
]
_digit_groups = [
    '',
    'thousand',
    'million',
    'billion',
    'trillion',
    'quadrillion',
]
_ordinal_suffixes = [
    ('one', 'first'),
    ('two', 'second'),
    ('three', 'third'),
    ('five', 'fifth'),
    ('eight', 'eighth'),
    ('nine', 'ninth'),
    ('twelve', 'twelfth'),
    ('ty', 'tieth'),
]

def _remove_commas(m):
    return m.group(1).replace(',', '')

def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')

def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    if dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    if cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    return 'zero dollars'

def _standard_number_to_words(n, digit_group):
    parts = []
    if n >= 1000:
        # Format next higher digit group.
        parts.append(_standard_number_to_words(n // 1000, digit_group + 1))
        n = n % 1000

    if n >= 100:
        parts.append('%s hundred' % _units[n // 100])
    if n % 100 >= len(_units):
        parts.append(_tens[(n % 100) // 10])
        parts.append(_units[(n % 100) % 10])
    else:
        parts.append(_units[n % 100])
    if n > 0:
        parts.append(_digit_groups[digit_group])
    return ' '.join([x for x in parts if x])

def _number_to_words(n):
    # Handle special cases first, then go to the standard case:
    if n >= 1000000000000000000:
        return str(n)  # Too large, just return the digits
    if n == 0:
        return 'zero'
    if n % 100 == 0 and n % 1000 != 0 and n < 3000:
        return _standard_number_to_words(n // 100, 0) + ' hundred'
    return _standard_number_to_words(n, 0)

def _expand_number(m):
    return _number_to_words(int(m.group(0)))

def _expand_ordinal(m):
    num = _number_to_words(int(m.group(1)))
    for suffix, replacement in _ordinal_suffixes:
        if num.endswith(suffix):
            return num[:-len(suffix)] + replacement
    return num + 'th'

def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text

# utils.audio.py
class AudioProcessor(object):
    def __init__(self,
                 sample_rate=None,
                 num_mels=None,
                 min_level_db=None,
                 frame_shift_ms=None,
                 frame_length_ms=None,
                 ref_level_db=None,
                 num_freq=None,
                 power=None,
                 preemphasis=0.0,
                 signal_norm=None,
                 symmetric_norm=None,
                 max_norm=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 clip_norm=True,
                 griffin_lim_iters=None,
                 do_trim_silence=False,
                 sound_norm=False,
                 **_):

        print("\t[-] Setting up Audio Processor...")

        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.min_level_db = min_level_db or 0
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.ref_level_db = ref_level_db
        self.num_freq = num_freq
        self.power = power
        self.preemphasis = preemphasis
        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.mel_fmin = mel_fmin or 0
        self.mel_fmax = mel_fmax
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.sound_norm = sound_norm
        self.n_fft, self.hop_length, self.win_length = self._stft_parameters()
        assert min_level_db != 0.0, " [!] min_level_db is 0"
        members = vars(self)
        for key, value in members.items():
            print("\t | > {}:{}".format(key, value))

    def save_wav(self, wav, path):
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        scipy.io.wavfile.write(path, self.sample_rate, wav_norm.astype(np.int16))

    def _linear_to_mel(self, spectrogram):
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _mel_to_linear(self, mel_spec):
        inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        return np.maximum(1e-10, np.dot(inv_mel_basis, mel_spec))

    def _build_mel_basis(self, ):
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2
        return librosa.filters.mel(
            self.sample_rate,
            self.n_fft,
            n_mels=self.num_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)

    def _normalize(self, S):
        """Put values in [0, self.max_norm] or [-self.max_norm, self.max_norm]"""
        #pylint: disable=no-else-return
        if self.signal_norm:
            S_norm = ((S - self.min_level_db) / - self.min_level_db)
            if self.symmetric_norm:
                S_norm = ((2 * self.max_norm) * S_norm) - self.max_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, -self.max_norm, self.max_norm)
                return S_norm
            else:
                S_norm = self.max_norm * S_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, 0, self.max_norm)
                return S_norm
        else:
            return S

    def _denormalize(self, S):
        """denormalize values"""
        #pylint: disable=no-else-return
        S_denorm = S
        if self.signal_norm:
            if self.symmetric_norm:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, -self.max_norm, self.max_norm)
                S_denorm = ((S_denorm + self.max_norm) * -self.min_level_db / (2 * self.max_norm)) + self.min_level_db
                return S_denorm
            else:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, 0, self.max_norm)
                S_denorm = (S_denorm * -self.min_level_db /
                            self.max_norm) + self.min_level_db
                return S_denorm
        else:
            return S

    def _stft_parameters(self, ):
        """Compute necessary stft parameters with given time values"""
        n_fft = (self.num_freq - 1) * 2
        factor = self.frame_length_ms / self.frame_shift_ms
        assert (factor).is_integer(), " [!] frame_shift_ms should divide frame_length_ms"
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(hop_length * factor)
        return n_fft, hop_length, win_length

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    @staticmethod
    def _db_to_amp(x):
        return np.power(10.0, x * 0.05)

    def apply_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" [!] Preemphasis is set 0.0.")
        return scipy.signal.lfilter([1, -self.preemphasis], [1], x)

    def apply_inv_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" [!] Preemphasis is set 0.0.")
        return scipy.signal.lfilter([1], [1, -self.preemphasis], x)

    def spectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db
        return self._normalize(S)

    def melspectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        return self._normalize(S)

    def inv_spectrogram(self, spectrogram):
        """Converts spectrogram to waveform using librosa"""
        S = self._denormalize(spectrogram)
        S = self._db_to_amp(S + self.ref_level_db)  # Convert back to linear
        # Reconstruct phase
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        return self._griffin_lim(S**self.power)

    def inv_mel_spectrogram(self, mel_spectrogram):
        '''Converts mel spectrogram to waveform using librosa'''
        D = self._denormalize(mel_spectrogram)
        S = self._db_to_amp(D + self.ref_level_db)
        S = self._mel_to_linear(S)  # Convert back to linear
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        return self._griffin_lim(S**self.power)

    def out_linear_to_mel(self, linear_spec):
        S = self._denormalize(linear_spec)
        S = self._db_to_amp(S + self.ref_level_db)
        S = self._linear_to_mel(np.abs(S))
        S = self._amp_to_db(S) - self.ref_level_db
        mel = self._normalize(S)
        return mel

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for _ in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode='constant'
        )

    def _istft(self, y):
        return librosa.istft(
            y, hop_length=self.hop_length, win_length=self.win_length)

    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        window_length = int(self.sample_rate * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(threshold_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x:x + window_length]) < threshold:
                return x + hop_length
        return len(wav)

    def trim_silence(self, wav):
        """ Trim silent parts with a threshold and 0.01 sec margin """
        margin = int(self.sample_rate * 0.01)
        wav = wav[margin:-margin]
        return librosa.effects.trim(
            wav, top_db=40, frame_length=self.win_length, hop_length=self.hop_length)[0]

    @staticmethod
    def mulaw_encode(wav, qc):
        mu = 2 ** qc - 1
        # wav_abs = np.minimum(np.abs(wav), 1.0)
        signal = np.sign(wav) * np.log(1 + mu * np.abs(wav)) / np.log(1. + mu)
        # Quantize signal to the specified number of levels.
        signal = (signal + 1) / 2 * mu + 0.5
        return np.floor(signal,)

    @staticmethod
    def mulaw_decode(wav, qc):
        """Recovers waveform from quantized values."""
        mu = 2 ** qc - 1
        x = np.sign(wav) / mu * ((1 + mu) ** np.abs(wav) - 1)
        return x

    def load_wav(self, filename, sr=None):
        if sr is None:
            x, sr = sf.read(filename)
        else:
            x, sr = librosa.load(filename, sr=sr)
        if self.do_trim_silence:
            try:
                x = self.trim_silence(x)
            except ValueError:
                print(f' [!] File cannot be trimmed for silence - {filename}')
        assert self.sample_rate == sr, "%s vs %s"%(self.sample_rate, sr)
        if self.sound_norm:
            x = x / abs(x).max() * 0.9
        return x

    @staticmethod
    def encode_16bits(x):
        return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

    @staticmethod
    def quantize(x, bits):
        return (x + 1.) * (2**bits - 1) / 2

    @staticmethod
    def dequantize(x, bits):
        return 2 * x / (2**bits - 1) - 1

# utils.generic_utils.py
def resource_path(relative_path):
    return os.path.join(os.environ['tts_base_dir'], relative_path)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

def load_config(config_path):
    config = AttrDict()
    with open(config_path, "r") as f:
        input_str = f.read()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)
    return config

def get_git_branch():
    try:
        out = subprocess.check_output(["git", "branch"]).decode("utf8")
        current = next(line for line in out.split("\n")
                       if line.startswith("*"))
        current.replace("* ", "")
    except subprocess.CalledProcessError:
        current = "inside_docker"
    return current

def get_commit_hash():
    """https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script"""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except subprocess.CalledProcessError:
        commit = "0000000"
    print('\t > Git Hash: {}'.format(commit))
    return commit

def create_experiment_folder(root_path, model_name, debug):
    """ Create a folder with the current date and time """
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    commit_hash = get_commit_hash()
    output_folder = os.path.join(root_path, model_name + '-' + date_str + '-' + commit_hash)
    os.makedirs(output_folder, exist_ok=True)
    print("\t > Experiment folder: {}".format(output_folder))
    return output_folder

def remove_experiment_folder(experiment_path):
    """Check folder if there is a checkpoint, otherwise remove the folder"""

    checkpoint_files = glob.glob(experiment_path + "/*.pth.tar")
    if not checkpoint_files:
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
            print("\t ! Run is removed from {}".format(experiment_path))
    else:
        print("\t ! Run is kept in {}".format(experiment_path))

def copy_config_file(config_file, out_path, new_fields):
    config_lines = open(config_file, "r").readlines()
    # add extra information fields
    for key, value in new_fields.items():
        if type(value) == str:
            new_line = '"{}":"{}",\n'.format(key, value)
        else:
            new_line = '"{}":{},\n'.format(key, value)
        config_lines.insert(1, new_line)
    config_out_file = open(out_path, "w")
    config_out_file.writelines(config_lines)
    config_out_file.close()

def _trim_model_state_dict(state_dict):
    r"""Remove 'module.' prefix from state dictionary. It is necessary as it
    is loded for the next time by model.load_state(). Otherwise, it complains
    about the torch.DataParallel()"""

    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def save_checkpoint(model, optimizer, optimizer_st, model_loss, out_path, current_step, epoch):
    checkpoint_path = 'checkpoint_{}.pth.tar'.format(current_step)
    checkpoint_path = os.path.join(out_path, checkpoint_path)
    print("\t | | > Checkpoint saving : {}".format(checkpoint_path))

    new_state_dict = model.state_dict()
    state = {
        'model': new_state_dict,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'step': current_step,
        'epoch': epoch,
        'linear_loss': model_loss,
        'date': datetime.date.today().strftime("%B %d, %Y"),
        'r': model.decoder.r
    }
    torch.save(state, checkpoint_path)

def save_best_model(model, optimizer, model_loss, best_loss, out_path, current_step, epoch):
    if model_loss < best_loss:
        new_state_dict = model.state_dict()
        state = {
            'model': new_state_dict,
            'optimizer': optimizer.state_dict(),
            'step': current_step,
            'epoch': epoch,
            'linear_loss': model_loss,
            'date': datetime.date.today().strftime("%B %d, %Y"),
            'r': model.decoder.r
        }
        best_loss = model_loss
        bestmodel_path = 'best_model.pth.tar'
        bestmodel_path = os.path.join(out_path, bestmodel_path)
        print("\n > BEST MODEL ({0:.5f}) : {1:}".format(
            model_loss, bestmodel_path))
        torch.save(state, bestmodel_path)
    return best_loss

def check_update(model, grad_clip, ignore_stopnet=False):
    r'''Check model gradient against unexpected jumps and failures'''
    skip_flag = False
    if ignore_stopnet:
        grad_norm = torch.nn.utils.clip_grad_norm_([param for name, param in model.named_parameters() if 'stopnet' not in name], grad_clip)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if np.isinf(grad_norm):
        print("\t | > Gradient is INF !!")
        skip_flag = True
    return grad_norm, skip_flag

def lr_decay(init_lr, global_step, warmup_steps):
    r'''from https://github.com/r9y9/tacotron_pytorch/blob/master/train.py'''
    warmup_steps = float(warmup_steps)
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(step * warmup_steps**-1.5, step**-0.5)
    return lr

def adam_weight_decay(optimizer):
    """
    Custom weight decay operation, not effecting grad values.
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            current_lr = group['lr']
            weight_decay = group['weight_decay']
            param.data = param.data.add(-weight_decay * group['lr'], param.data)
    return optimizer, current_lr

def set_weight_decay(model, weight_decay, skip_list={"decoder.attention.v", "rnn", "lstm", "gru", "embedding"}):
    """
    Skip biases, BatchNorm parameters, rnns.
    and attention projection layer v
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or any([skip_name in name for skip_name in skip_list]):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{
        'params': no_decay,
        'weight_decay': 0.
    }, {
        'params': decay,
        'weight_decay': weight_decay
    }]

class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=0.1, last_epoch=-1):
        self.warmup_steps = float(warmup_steps)
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        return [
            base_lr * self.warmup_steps**0.5 *
            min(step * self.warmup_steps**-1.5, step**-0.5)
            for base_lr in self.base_lrs
        ]

def mk_decay(init_mk, max_epoch, n_epoch):
    return init_mk * ((max_epoch - n_epoch) / max_epoch)

def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (
        sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    # B x T_max
    return seq_range_expand < seq_length_expand

def set_init_dict(model_dict, checkpoint, c):
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k, v in checkpoint['model'].items():
        if k not in model_dict:
            print("\t | > Layer missing in the model definition: {}".format(k))
    # 1. filter out unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in checkpoint['model'].items() if k in model_dict
    }
    # 2. filter out different size layers
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if v.numel() == model_dict[k].numel()
    }
    # 3. skip reinit layers
    if c.reinit_layers is not None:
        for reinit_layer_name in c.reinit_layers:
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if reinit_layer_name not in k
            }
    # 4. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    print("\t | > {} / {} layers are restored.".format(len(pretrained_dict),
                                                     len(model_dict)))
    return model_dict

def setup_model(num_chars, num_speakers, c):
    print("\t[-] Using model: {}".format(c.model))
    model = Tacotron2(num_chars=num_chars,
                    num_speakers=num_speakers,
                    r=c.r,
                    postnet_output_dim=c.audio['num_mels'],
                    decoder_output_dim=c.audio['num_mels'],
                    attn_type=c.attention_type,
                    attn_win=c.windowing,
                    attn_norm=c.attention_norm,
                    prenet_type=c.prenet_type,
                    prenet_dropout=c.prenet_dropout,
                    forward_attn=c.use_forward_attn,
                    trans_agent=c.transition_agent,
                    forward_attn_mask=c.forward_attn_mask,
                    location_attn=c.location_attn,
                    attn_K=c.attention_heads,
                    separate_stopnet=c.separate_stopnet,
                    bidirectional_decoder=c.bidirectional_decoder)
    return model

def split_dataset(items):
    is_multi_speaker = False
    speakers = [item[-1] for item in items]
    is_multi_speaker = len(set(speakers)) > 1
    eval_split_size = 500 if len(items) * 0.01 > 500 else int(
        len(items) * 0.01)
    np.random.seed(0)
    np.random.shuffle(items)
    if is_multi_speaker:
        items_eval = []
        # most stupid code ever -- Fix it !
        while len(items_eval) < eval_split_size:
            speakers = [item[-1] for item in items]
            speaker_counter = collections.Counter(speakers)
            item_idx = np.random.randint(0, len(items))
            if speaker_counter[items[item_idx][-1]] > 1:
                items_eval.append(items[item_idx])
                del items[item_idx]
        return items_eval, items
    else:
        return items[:eval_split_size], items[eval_split_size:]

def gradual_training_scheduler(global_step, config):
    """Setup the gradual training schedule wrt number
    of active GPUs"""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        num_gpus = 1
    new_values = None
    # we set the scheduling wrt num_gpus
    for values in config.gradual_training:
        if global_step * num_gpus >= values[0]:
            new_values = values
    return new_values[1], new_values[2]

class KeepAverage():
    def __init__(self):
        self.avg_values = {}
        self.iters = {}

    def __getitem__(self, key):
        return self.avg_values[key]

    def add_value(self, name, init_val=0, init_iter=0):
        self.avg_values[name] = init_val
        self.iters[name] = init_iter

    def update_value(self, name, value, weighted_avg=False):
        if weighted_avg:
            self.avg_values[name] = 0.99 * self.avg_values[name] + 0.01 * value
            self.iters[name] += 1
        else:
            self.avg_values[name] = self.avg_values[name] * \
                self.iters[name] + value
            self.iters[name] += 1
            self.avg_values[name] /= self.iters[name]

    def add_values(self, name_dict):
        for key, value in name_dict.items():
            self.add_value(key, init_val=value)

    def update_values(self, value_dict):
        for key, value in value_dict.items():
            self.update_value(key, value)

# utils.speakers.py
def make_speakers_json_path(out_path):
    """Returns conventional speakers.json location."""
    return os.path.join(out_path, "speakers.json")

def load_speaker_mapping(out_path):
    """Loads speaker mapping if already present."""
    try:
        with open(make_speakers_json_path(out_path)) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_speaker_mapping(out_path, speaker_mapping):
    """Saves speaker mapping if not yet present."""
    speakers_json_path = make_speakers_json_path(out_path)
    with open(speakers_json_path, "w") as f:
        json.dump(speaker_mapping, f, indent=4)

def get_speakers(items):
    """Returns a sorted, unique list of speakers in a given dataset."""
    speakers = {e[2] for e in items}
    return sorted(speakers)

# utils.data.py
def _pad_data(x, length):
    _pad = 0
    assert x.ndim == 1
    return np.pad(
        x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_tensor(x, length):
    _pad = 0
    assert x.ndim == 2
    x = np.pad(
        x, [[0, 0], [0, length - x.shape[1]]],
        mode='constant',
        constant_values=_pad)
    return x

def prepare_tensor(inputs, out_steps):
    max_len = max((x.shape[1] for x in inputs))
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_tensor(x, pad_len) for x in inputs])

def _pad_stop_target(x, length):
    _pad = 1.
    assert x.ndim == 1
    return np.pad(
        x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def prepare_stop_target(inputs, out_steps):
    """ Pad row vectors with 1. """
    max_len = max((x.shape[0] for x in inputs))
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_stop_target(x, pad_len) for x in inputs])

def pad_per_step(inputs, pad_len):
    return np.pad(
        inputs, [[0, 0], [0, 0], [0, pad_len]],
        mode='constant',
        constant_values=0.0)

# utils.synthesis.py
def text_to_seqvec(text, CONFIG, use_cuda):
    text_cleaner = [CONFIG.text_cleaner]
    # text ot phonemes to sequence vector
    if CONFIG.use_phonemes:
        seq = np.asarray(
            phoneme_to_sequence(text, text_cleaner, CONFIG.phoneme_language,
                                CONFIG.enable_eos_bos_chars),
            dtype=np.int32)
    else:
        seq = np.asarray(text_to_sequence(text, text_cleaner), dtype=np.int32)
    # torch tensor
    chars_var = torch.from_numpy(seq).unsqueeze(0)
    if use_cuda:
        chars_var = chars_var.cuda()
    return chars_var.long()

def compute_style_mel(style_wav, ap, use_cuda):
    print(style_wav)
    style_mel = torch.FloatTensor(ap.melspectrogram(
        ap.load_wav(style_wav))).unsqueeze(0)
    if use_cuda:
        return style_mel.cuda()
    return style_mel

def run_model(model, inputs, CONFIG, truncated, speaker_id=None, style_mel=None):
    if CONFIG.use_gst:
        decoder_output, postnet_output, alignments, stop_tokens = model.inference(
            inputs, style_mel=style_mel, speaker_ids=speaker_id)
    else:
        if truncated:
            decoder_output, postnet_output, alignments, stop_tokens = model.inference_truncated(
                inputs, speaker_ids=speaker_id)
        else:
            decoder_output, postnet_output, alignments, stop_tokens = model.inference(
                inputs, speaker_ids=speaker_id)
    return decoder_output, postnet_output, alignments, stop_tokens

def parse_outputs(postnet_output, decoder_output, alignments):
    postnet_output = postnet_output[0].data.cpu().numpy()
    decoder_output = decoder_output[0].data.cpu().numpy()
    alignment = alignments[0].cpu().data.numpy()
    return postnet_output, decoder_output, alignment

def trim_silence(wav, ap):
    return wav[:ap.find_endpoint(wav)]

def inv_spectrogram(postnet_output, ap, CONFIG):
    if CONFIG.model in ["Tacotron", "TacotronGST"]:
        wav = ap.inv_spectrogram(postnet_output.T)
    else:
        wav = ap.inv_mel_spectrogram(postnet_output.T)
    return wav

def id_to_torch(speaker_id):
    if speaker_id is not None:
        speaker_id = np.asarray(speaker_id)
        speaker_id = torch.from_numpy(speaker_id).unsqueeze(0)
    return speaker_id

def synthesis(model,
              text,
              CONFIG,
              use_cuda,
              ap,
              speaker_id=None,
              style_wav=None,
              truncated=False,
              enable_eos_bos_chars=False, #pylint: disable=unused-argument
              do_trim_silence=False):
    """Synthesize voice for the given text.

        Args:
            model (TTS.models): model to synthesize.
            text (str): target text
            CONFIG (dict): config dictionary to be loaded from config.json.
            use_cuda (bool): enable cuda.
            ap (TTS.utils.audio.AudioProcessor): audio processor to process
                model outputs.
            speaker_id (int): id of speaker
            style_wav (str): Uses for style embedding of GST.
            truncated (bool): keep model states after inference. It can be used
                for continuous inference at long texts.
            enable_eos_bos_chars (bool): enable special chars for end of sentence and start of sentence.
            do_trim_silence (bool): trim silence after synthesis.
    """
    # GST processing
    style_mel = None
    if CONFIG.model == "TacotronGST" and style_wav is not None:
        style_mel = compute_style_mel(style_wav, ap, use_cuda)
    # preprocess the given text
    inputs = text_to_seqvec(text, CONFIG, use_cuda)
    speaker_id = id_to_torch(speaker_id)
    if speaker_id is not None and use_cuda:
        speaker_id = speaker_id.cuda()
    # synthesize voice
    decoder_output, postnet_output, alignments, stop_tokens = run_model(
        model, inputs, CONFIG, truncated, speaker_id, style_mel)
    # convert outputs to numpy
    postnet_output, decoder_output, alignment = parse_outputs(
        postnet_output, decoder_output, alignments)
    # plot results
    wav = inv_spectrogram(postnet_output, ap, CONFIG)
    # trim silence
    if do_trim_silence:
        wav = trim_silence(wav, ap)
    return wav, alignment, decoder_output, postnet_output, stop_tokens
