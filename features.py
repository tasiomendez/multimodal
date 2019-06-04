import logging
import os
import pandas as pd
from pyAudioAnalysis import audioBasicIO, audioFeatureExtraction
from pyAudioAnalysis import audioTrainTest as aT
from .video import VideoAnalyzer

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)

path = os.path.dirname(os.path.abspath(__file__))


class Features:
    """
        Extract features from different sources base class
    """

    def __init__(self, verbose=True, **kwargs):
        self.verbose = verbose

    def run(self):
        pass


class AudioFeatures(Features):
    """
        Extract features from audio file
        It computes 34 of short-term features implemented in pyAudioAnalysis library.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, file, window=0.050, step=0.025):
        logger.info('Extracting features from audio file {}...'.format(os.path.basename(file)))
        df = self.features(file, window, step).mean()
        file, arousal, valence = self.get_arousal_valence(file)
        df['arousal'] = arousal
        df['valence'] = valence
        return df

    def features(self, file, window=0.050, step=0.025):
        [Fs, x] = audioBasicIO.readAudioFile(file)
        F, f_names = audioFeatureExtraction.stFeatureExtraction(audioBasicIO.stereo2mono(x), Fs, window*Fs, step*Fs)
        return pd.DataFrame(data=F.T, columns=f_names)

    def get_arousal_valence(self, file):
        """Returns file, arousal and valence is this order."""
        try:
            values, names = aT.fileRegression(file, 'models/svmSpeechEmotion', 'svm')
            return os.path.basename(file).split('.')[0], values[0], values[1]
        except Exception:
            return os.path.basename(file).split('.')[0], float('NaN'), float('NaN')


class VideoFeatures(Features):
    """
        Extract features from audio file
        It a pretrained deep learning model for computing emotions.
    """

    def __init__(self, **kwargs):
        self.model = os.path.join(path, 'models/haarcascade_frontalface_default.xml')
        super().__init__(**kwargs)

    def run(self, file):
        logger.info('Extracting features from video file {}...'.format(os.path.basename(file)))
        return VideoAnalyzer(self.model, file).analyze().toDataFrame()

    def synchronize(self, df, start, end):
        bounds = pd.Series({'start': start, 'end': end})
        if end:
            emotions = df[(df.timestamp >= start) & (df.timestamp < end)].mean()
        else:
            emotions = df[(df.timestamp >= start)].mean()
        # Return a pd.Series
        return pd.concat([bounds, emotions]).drop('timestamp')


class BimodalFeatures(Features):
    """
        Extract features from two sources and synchronize them.

        :param file: DataFrame with the bounds for synchronization. It must contain
                     at least #starttime and #endtime columns of each chunk.

        For accomplish an analysis, the following requirements must be met.
        - One audio file per utterance
        - One video file
        - A DataFrame with transcriptions whose column must be named 'transcription'
    """

    def __init__(self, df, **kwargs):
        if not '#starttime' in df.columns or not '#endtime' in df.columns:
            raise Exception('Some columns are missing in bounds file')
        self.bounds = df[['#starttime', '#endtime']]
        logger.info('Bounds file configured succesfully!')
        super().__init__(**kwargs)

    def run_audio(self, files):
        if len(files) != len(self.bounds):
            raise Exception('{} audio files are needed and {} were provided'.format(len(self.bounds), len(files)))
        series = []
        for file in files:
            series.append(AudioFeatures().run(file))
        self.audio = pd.concat([serie for serie in series], axis=1).transpose()
        return self

    def run_video(self, file):
        series = []
        features = VideoFeatures().run(file)
        for start, end in zip(self.bounds['#starttime'], self.bounds['#endtime']):
            series.append(VideoFeatures().synchronize(features, start, end))
        self.video = pd.concat([serie for serie in series], axis=1).transpose()
        return self

    def run_text(self, df):
        self.text = df['transcription']
        return self

    def run(self, modalities, text=None, audio=None, video=None):
        if len(modalities) != 2:
            raise Exception('length of modalities is {} and must be 2'.format(len(modalities)))
        if 'text' in modalities and 'audio' in modalities:
            self.run_text(text).run_audio(audio)
            return pd.concat([self.text, self.audio], axis=1)
        elif 'audio' in modalities and 'video' in modalities:
            self.run_audio(audio).run_video(video)
            return pd.concat([self.audio, self.video], axis=1)
        elif 'video' in modalities and 'text' in modalities:
            self.run_video(video).run_text(text)
            return pd.concat([self.video, self.text], axis=1)
        else:
            raise Exception('This combination of modalities are not supported!')


class MultimodalFeatures(BimodalFeatures):
    """
        Extract features from three sources and synchronize them.

        :param file: DataFrame with the bounds for synchronization. It must contain
                     at least #starttime and #endtime columns of each chunk.

        For accomplish an analysis, the following requirements must be met.
        - One audio file per utterance
        - One video file
        - A DataFrame with transcriptions whose column must be named 'transcription'
    """

    def __init__(self, df, **kwargs):
        super().__init__(df, **kwargs)

    def run(self, text=None, audio=None, video=None):
        self.run_text(text).run_audio(audio).run_video(video)
        return pd.concat([self.text, self.audio, self.video], axis=1)
