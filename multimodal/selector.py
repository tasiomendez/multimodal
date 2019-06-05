import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class Selector:
    """Features selector for diferent features based on data."""

    def __init__(self):
        pass

    @staticmethod
    def pca(df):
        """PCA selector for audio features"""
        columns = ['zcr', 'spectral_centroid', 'spectral_entropy', 'spectral_flux', 'spectral_rolloff']
        return df[columns]

    @staticmethod
    def text(df):
        """Text features selector"""
        return df['transcription']

    @staticmethod
    def audio(df):
        """Audio features selector"""
        columns = ['zcr','energy','energy_entropy','spectral_centroid','spectral_spread','spectral_entropy','spectral_flux',
                   'spectral_rolloff','mfcc_1','mfcc_2','mfcc_3','mfcc_4','mfcc_5','mfcc_6','mfcc_7','mfcc_8','mfcc_9',
                   'mfcc_10','mfcc_11','mfcc_12','mfcc_13','chroma_1','chroma_2','chroma_3','chroma_4','chroma_5','chroma_6',
                   'chroma_7','chroma_8','chroma_9','chroma_10','chroma_11','chroma_12','chroma_std','arousal','valence']
        return df[columns]

    @staticmethod
    def video(df):
        """Video features selector"""
        columns = ['anger','calm','happiness']
        return df[columns]
