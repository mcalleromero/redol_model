import pandas as pd
import numpy as np

import random

import util.generator as generator

def print_nclasses(path):
    df = pd.read_csv(path)

    print(df['class'].value_counts())

def generate_modified_dataset(path, model='twonorm', proportion=.5, num_examples=5000, dimm=21):
    num_zeros_title = proportion
    num_unos_title = round(1 - num_zeros_title, 1)
    num_examples = num_examples * 2

    print(f'Creating dataset with:')
    print(f'\tModel: {model}')
    print(f'\tzeros: {num_zeros_title}%')
    print(f'\tunos: {num_unos_title}%')
    print(f'\tDimms: {dimm}')
    print(f'\tNumber of instances: {int(num_examples/2)}')
    X, y = generator.create_full_dataset(num_examples, dimm, model)

    df = pd.DataFrame(X)
    df['class'] = y

    indexes_zeros_arr = np.arange(df[df['class'] == 0.0].shape[0])
    indexes_unos_arr = np.arange(num_examples - df[df['class'] == 0.0].shape[0], num_examples)

    random.seed(4)
    random.shuffle(indexes_zeros_arr)
    random.shuffle(indexes_unos_arr)

    num_of_zeros = int((num_examples / 2) * proportion)
    num_of_unos = int((num_examples / 2) - num_of_zeros)

    print(f'\tNumber of zeros: {num_of_zeros}')
    print(f'\tNumber of unos: {num_of_unos}')

    final_zeros = indexes_zeros_arr[:num_of_zeros]
    final_unos = indexes_unos_arr[:num_of_unos]

    final_df = df[df['class'] == 0.0].loc[final_zeros].append(df[df['class'] == 1.0].loc[final_unos])

    title = f'zeros{num_zeros_title}_unos{num_unos_title}_{model}_{dimm}dimm.csv'

    print(f'Generating {title}')
    final_df.to_csv(f'{path}/{title}', index=False, header=list(map(lambda x: f'f{x}', np.arange(dimm)))+['class'])