from imports import import_all
import_all()

import warnings
warnings.simplefilter('ignore')

import torch
from utils import init_jit_model, read_batch, prepare_model_input

device = torch.device('cpu')

main_folder = 'ua-silero-demo'
rec_folder  = 'recordings'
file_name   = 'spravy.wav'

full_path   = f'./{main_folder}/{rec_folder}/{file_name}'


def read_file(file_path):
    jit_model = f'./{main_folder}/model/ua_v3_jit.model'
    model, decoder = init_jit_model(jit_model, device=device)

    test_files = [file_path]
    res = ''
    model_input = prepare_model_input(read_batch(test_files), device=device)
    output = model(model_input)
    for example in output:
        res += decoder(example.cpu())
    return res


lol = read_file(full_path)



print(lol)