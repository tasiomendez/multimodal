import logging
import os
import pandas as pd
from .utils import custom_tokenizer
from .selector import Selector
from sklearn.externals import joblib

# Models
# pipeline = joblib.load('models/text_audio.pkl')
# pipeline = joblib.load('models/audio_video.pkl')
# pipeline = joblib.load('models/video_text.pkl')
# pipeline = joblib.load('models/multimodal.pkl')

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)

path = os.path.dirname(os.path.abspath(__file__))


def text_analyzer(text):
    """
        Pretrained model for emotion recognition in text.
        :param text: text to predict
    """
    pipeline = joblib.load(os.path.join(path, 'models/text.pkl'))
    return pipeline.predict([text])[0]


def audio_analyzer(audio):
    """
        Pretrained model for emotion recognition in audio.
        :param audio: audio features as a pd.Series
    """
    pipeline = joblib.load(os.path.join(path, 'models/audio.pkl'))
    features = pd.DataFrame(audio).T
    return pipeline.predict(features)


def video_analyzer(video):
    """
        Pretrained model for emotion recognition in audio.
        :param video: video features as a pd.Series
    """
    pipeline =  joblib.load(os.path.join(path, 'models/video.pkl'))
    features = pd.DataFrame(video).T
    return pipeline.predict(features)
