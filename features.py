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
        It computes 34 of short-term features implemented in pyAudioAnalysis library.
    """

    def __init__(self, **kwargs):
        self.model = os.path.join(path, 'models/haarcascade_frontalface_default.xml')
        super().__init__(**kwargs)

    def run(self, file):
        return VideoAnalyzer(self.model, file).analyze().toDataFrame()
