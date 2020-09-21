from pathlib import Path
from argparse import ArgumentParser
from random import uniform, seed, choice, randint, random

from AudioFile import AudioFile

SEED = 1234  # for reproducibility


def augment(input_dir: Path, noise_dir: Path, output_dir: Path):
    seed(SEED)

    if not input_dir.exists:
        raise Exception('Input directory does not exist.')

    if not noise_dir.exists:
        raise Exception('Noise directory does not exist.')

    if not output_dir.exists:
        print("Making output directory {}".format(output_dir))
        output_dir.mkdir(parents=True)

    filelist = [x for x in input_dir.glob('*.wav')]
    print("{} input files found".format(len(filelist)))

    noiselist = set([x for x in noise_dir.glob('*.wav')])
    print("{} noise files found".format(len(noiselist)))

    irs = [x for x in Path('./IMreverbs').glob('*.wav')]

    while len(filelist) > 0:
        print("{} files remaining...".format(len(filelist)))
        f1 = filelist.pop()
        print("Choosing noise file...")
        noise = choice(tuple(noiselist))

        # load audio files and apply a random amount of processing to noisy file:
        #   gain reduction in steps of -6 db from [-6 db, 0 db]
        #   varispeed between [0.9, 1.1]
        #   start position of audio in noise file (trimming start of file)
        reduction = [0, -3, -6]
        print("Loading files and modify noise file...")
        f1 = AudioFile(path=f1)
        noise = AudioFile(path=noise) \
            .varispeed(uniform(0.9, 1.1)) \
            .gain(choice(reduction)) \
            .trim_start(relative_start=uniform(0.0, 0.5))

        # add dynamic lpf to simulate speaker turning away
        print("Applying dynamic lpf...")
        filter_start = random()
        filter_end = random()
        if filter_start < filter_end:
            f1.dynamic_lpf(cutoff=uniform(1000, 8000),
                           order=randint(0, 3),
                           relative_start=random(),
                           relative_end=random(),
                           exponential=random())

        # add noise to audio
        print("Mixing clean audio and noise...")
        f1.mix(noise, maintain_length=True)

        # choose random impulse response and add reverb to noisy audio
        print("Applying convolution reverb...")
        ir = AudioFile(path=choice(irs))
        f1.conv_reverb(ir, wet_db=uniform(-50, 10), predelay=uniform(0, 50))

        # filtering
        print("Applying static LPF and HPF...")
        f1.lpf(uniform(5000, 8000))
        f1.hpf(uniform(0, 250))

        # clipping
        print("Applying digital clipping...")
        clipping = [0.0, 1.0, 2.0, 3.0]
        f1.clip(choice(clipping))

        # save
        print("Saving file...")
        f1.save(output_path=output_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help='input directory of files to use for augmentation')
    parser.add_argument('-n', '--noise_dir', type=str, help='input directory of noise to use for augmentation')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory for augmented files')
    args = parser.parse_args()
    augment(Path(args.input_dir), Path(args.noise_dir), Path(args.output_dir))
