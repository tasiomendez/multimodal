import logging
import os
import pandas as pd
from .utils import custom_tokenizer
from .selector import Selector
from sklearn.externals import joblib

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
    pipeline = joblib.load(os.path.join(path, 'models/video.pkl'))
    features = pd.DataFrame(video).T
    return pipeline.predict(features)


def bimodal_analyzer(features, modalities):
    """
        Pretrained model for emotion recognition in two modalities.
        :param features: features as a pd.DataFrame
        :param modalities: tuple with modalities

        **Example**::

            modalities = ('audio', 'video')
            prediction = bi_analyzer(features, modalities)
    """
    if len(modalities) != 2:
        raise Exception('length of modalities is {} and must be 2'.format(len(modalities)))
    if 'text' in modalities and 'audio' in modalities:
        pipeline = joblib.load(os.path.join(path, 'models/text_audio.pkl'))
    elif 'audio' in modalities and 'video' in modalities:
        pipeline = joblib.load(os.path.join(path, 'models/audio_video.pkl'))
    elif 'video' in modalities and 'text' in modalities:
        pipeline = joblib.load(os.path.join(path, 'models/video_text.pkl'))
    else:
        raise Exception('This combination of modalities are not supported!')
    return pipeline.predict(features)


def multimodal(features):
    """
        Pretrained model for emotion recognition in text + audio + video.
        :param features: text + audio + video features as a pd.DataFrame
    """
    pipeline =  joblib.load(os.path.join(path, 'models/multimodal.pkl'))
    return pipeline.predict(features)
