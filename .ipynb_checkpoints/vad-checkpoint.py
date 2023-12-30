SAMPLING_RATE = 16000

import torch
torch.set_num_threads(1)

#from IPython.display import Audio
from pprint import pprint
# download example
#torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')

USE_ONNX = False # change this to True if you want to test onnx model
  
model, utils = torch.hub.load(repo_or_dir='./snakers4_silero-vad_master',
                              source='local',
                              model='silero_vad',
                              force_reload=True,
                              onnx=USE_ONNX)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils



wav = read_audio('en_example.wav', sampling_rate=SAMPLING_RATE)
# get speech timestamps from full audio file
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
pprint(speech_timestamps)