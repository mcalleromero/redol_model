import sys
sys.path.append('/home/mario.calle/master/redol_model/')

from util.generate_modified import generate_modified_dataset

def main():
    generate_modified_dataset(path='../data/modified', model='ringnorm_normal', proportion=.1, num_examples=5000, dimm=20)
    generate_modified_dataset(path='../data/modified', model='ringnorm_normal', proportion=.2, num_examples=5000, dimm=20)
    generate_modified_dataset(path='../data/modified', model='ringnorm_normal', proportion=.3, num_examples=5000, dimm=20)
    generate_modified_dataset(path='../data/modified', model='ringnorm_normal', proportion=.4, num_examples=5000, dimm=20)
    generate_modified_dataset(path='../data/modified', model='ringnorm_normal', proportion=.5, num_examples=5000, dimm=20)
    generate_modified_dataset(path='../data/modified', model='ringnorm_normal', proportion=.6, num_examples=5000, dimm=20)
    generate_modified_dataset(path='../data/modified', model='ringnorm_normal', proportion=.7, num_examples=5000, dimm=20)
    generate_modified_dataset(path='../data/modified', model='ringnorm_normal', proportion=.8, num_examples=5000, dimm=20)
    generate_modified_dataset(path='../data/modified', model='ringnorm_normal', proportion=.9, num_examples=5000, dimm=20)


if __name__ == "__main__":
    main()