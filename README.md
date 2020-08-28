# AudioTransform

AudioTransform is a package for modifying mono audio files for use in augmenting datasets of audio.

Operations:
* load/save audio data
* resample
* scale amplitude
* normalize amplitude
* digitally clip amplitude
* pad with silence
* varispeed
* mix two audio files
* low-pass filter
* convolution reverb

# Installation

pip install -r requirements.txt

# How To

See `AugmentAudio.ipynb` for examples of how to process audio.

Note - transformations are applied to object itself allowing method chaining for complex signal chains.

If a transformation is only being applied temporarily (e.g. scaling amplitude during mixing) use the copy() method to create a new audio file to transform

Example:

```
audio1 = AudioFile('example1.wav')
audio2 = AudioFile('example2.wav')

# audio1 is mixed with (audio2 * 0.5) - after this operation, audio2 is now scaled by 0.5
audio1.mix(audio2.scale(0.5, relative_start=0.0))

# The following line will mix audio1 with (audio2 * 0.5) - audio2 remains unchanged by this operation
audio1.mix(audio2.copy.scale(0.5), relative_start=0.0))
```

# TO-DO

* Stereo file processing
* Dynamic filtering
