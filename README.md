## Multimodal Analysis

This project allows the user to perform a multimodal analysis for emotion recognition in Spanish-language videos. It uses pre-trained models to predict emotions from different data sources.

The models have been evaluated using a cross validation with a maximum accuracy of 0.8014 when the three modalities (text, audio, video) are combined.

### Configuration

This tool can be used to analyze a full video. However, it is needed to follow several rules so the script can read the video files in the right way in order to synchronize them.

- Each video must be divided in X utterances.
- A bounds file which indicates where an utterance start and end. This file must be a CSV with at least two columns: `#starttime` and `#endtime`.
- The text file must have one row per utterance in CSV format.
- One audio file is needed for each utterance in WAV format.

The script can be executed from the command line using a configuration file or passing the arguments through the terminal directly. The available options are the following.

| Option | Description |
| --- | --- |
| `--file FILE` | Path of the configuration file, if this option is used, any other one can be used. |
| `--bounds BOUNDS` | Bounds path file |
| `--text TEXT` | Text path file |
| `--audio AUDIO [AUDIO ...]` | Audio files for each utterance |
| `--audio-dir AUDIO_DIR` | Audio directory where the audio files. This option is exclusive with the previous one. |
| `--video VIDEO` | Video file path |

> The options for the configuration file are the same that are described in the table. You can see an example in the `config.yml` file.

Once the script is executed, a summarizing table is presented indicating for each utterance if the emotion is positive or negative. The analysis accomplished depends on the sources input. Thus, all the possible options will be computed.

### Import multimodal

This project can be also be used to analyze a video manually. For this purpose, two different tools are provided: one for extracting features from the different input sources and one for analyzing the features extracted.

The FeaturesExtractor can be used to extract features individually for each sources or synchronize them in pairs or all of them. The AudioFeature extractor computes 34 of short-term features implemented in [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) library. The VideoFeature extractor uses a pretrained deep learning model for computing emotions.

```python
from multimodal import features

# AudioFeatures extractor
ft = features.AudioFeatures().run('path/to/file.wav')

# VideoFeatures extractor
ft = features.VideoFeatures().run('path/to/file.mp4')

# BimodalFeatures extractor
bf = features.BimodalFeatures('path/to/bounds.csv')
sources = ('audio', 'text')
ft = bf.run(sources, audio=['path/to/audio.wav', ...], text='path/to/text.csv')

# MultimodalFeatures extractor
mf = features.MultimodalFeatures('path/to/bounds.csv')
ft = mf.run(video='path/to/video.mp4', audio=['path/to/audio.wav', ...], text='path/to/text.csv')
```
