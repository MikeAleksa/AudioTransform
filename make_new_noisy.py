from pathlib import Path
from argparse import ArgumentParser

from AudioFile import AudioFile as af

SEED = 1234


def augment(input_dir: Path, output_dir: Path):
    # load audio file
    # make a copy
    # add dynamic lpf to simulate speaker turning away
    # add noise (mix)
    # add reverb
    # filtering
    # clipping


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help='input directory of files to use for augmentation')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory for augmented files')
    args = parser.parse_args()
    augment(Path(args.input_dir), Path(args.output_dir))
