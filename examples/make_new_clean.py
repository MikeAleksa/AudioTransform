from pathlib import Path
from argparse import ArgumentParser
from random import uniform, seed, choice

from AudioFile import AudioFile

SEED = 1234  # for reproducibility


def augment(input_dir: Path, output_dir: Path):
    seed(SEED)

    if not input_dir.exists:
        raise Exception('Input directory does not exist.')

    if not output_dir.exists:
        print("Making output directory {}".format(output_dir))
        output_dir.mkdir(parents=True)

    filelist = set([x for x in input_dir.glob('*.wav')])
    print("{} input files found".format(len(filelist)))

    while len(filelist) > 1:
        print("{} files remaining...".format(len(filelist)))
        f1 = choice(tuple(filelist))
        filelist.remove(f1)

        f2 = choice(tuple(filelist))
        filelist.remove(f2)

        # load audio files and apply a random amount of:
        #   gain reduction in steps of -3 db from [-15 db, 0 db]
        #   varispeed between [0.9, 1.1]
        reduction = [0, -3, -6, -12, -15]
        f1 = AudioFile(path=f1).varispeed(uniform(0.9, 1.1)).gain(choice(reduction))
        f2 = AudioFile(path=f2).varispeed(uniform(0.9, 1.1)).gain(choice(reduction))

        # mix two audio files - random amount of overlap from [0.5 to 1.5]
        f1.mix(audio=f2, relative_start=uniform(0.5, 1.5))

        # add a random amounts of silence [0 sec, 5 sec] before and after audio
        f1.add_silence(sec_before=uniform(0, 5), sec_after=uniform(0, 5))

        # save as new clean file
        filename = f1.filename.split(".wav")[0] + "_" + f2.filename
        f1.save(output_path=output_dir, filename=filename)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help='input directory of files to use for augmentation')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory for augmented files')
    args = parser.parse_args()
    augment(Path(args.input_dir), Path(args.output_dir))
