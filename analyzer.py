import logging
import os
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


class Analyzer:

    def __init__(self, verbose=True, **kwargs):
        self.verbose = verbose

    @staticmethod
    def text(text):
        pipeline = joblib.load(os.path.join(path, 'models/text.pkl'))
        return pipeline.predict([text])[0]

    @staticmethod
    def audio(audio):
        pipeline = joblib.load('models/audio.pkl')
        pass

    @staticmethod
    def video(video):
        pipeline =  joblib.load('models/video.pkl')
        pass
