import pandas as pd
import numpy as np

import random

import util.generator as generator

def print_nclasses(path):
    df = pd.read_csv(path)

    print(df['class'].value_counts())

def generate_modified_dataset(path, model='twonorm', proportion=.5, num_examples=5000, dimm=21):
    num_zeroes_title = proportion
    num_unoes_title = 1 - num_zeroes_title
    num_examples = num_examples * 2

    print(f'Creating dataset with:')
    print(f'\tModel: {model}')
    print(f'\tZeroes: {num_zeroes_title}%')
    print(f'\tUnoes: {num_unoes_title}%')
    print(f'\tDimms: {dimm}')
    print(f'\tNumber of instances: {int(num_examples/2)}')
    X, y = generator.create_full_dataset(num_examples, dimm, model)

    df = pd.DataFrame(X)
    df['class'] = y

    indexes_zeroes_arr = np.arange(df[df['class'] == 0.0].shape[0])
    indexes_unoes_arr = np.arange(num_examples - df[df['class'] == 0.0].shape[0], num_examples)

    random.seed(4)
    random.shuffle(indexes_zeroes_arr)
    random.shuffle(indexes_unoes_arr)

    num_of_zeroes = int((num_examples / 2) * proportion)
    num_of_unoes = int((num_examples / 2) - num_of_zeroes)

    final_zeroes = indexes_zeroes_arr[:num_of_zeroes]
    final_unoes = indexes_unoes_arr[:num_of_unoes]

    final_df = df[df['class'] == 0.0].loc[final_zeroes].append(df[df['class'] == 1.0].loc[final_unoes])

    title = f'zeroes{num_zeroes_title}_unoes{num_unoes_title}_{model}.csv'

    print(f'Generating {title}')
    final_df.to_csv(f'{path}/{title}', index=False, header=list(map(lambda x: f'f{x}', np.arange(dimm)))+['class'])
