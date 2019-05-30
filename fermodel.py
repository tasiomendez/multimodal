import cv2
import logging
import numpy as np
import json
from keras.models import load_model
from pkg_resources import resource_filename
from scipy import misc

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)

class FERModel:
    """
    Pretrained deep learning model for facial expression recognition.

    :param target_emotions: set of target emotions to classify
    :param verbose: if true, will print out extra process information

    **Example**::

        from fermodel import FERModel

        target_emotions = ['happiness', 'disgust', 'surprise']
        model = FERModel(target_emotions, verbose=True)

    """

    POSSIBLE_EMOTIONS = ['anger', 'fear', 'calm', 'sadness', 'happiness', 'surprise', 'disgust']

    def __init__(self, target_emotions, verbose=False):
        self.target_emotions = target_emotions
        self.emotion_index_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'happiness': 3,
            'sadness': 4,
            'surprise': 5,
            'calm': 6
        }
        self._check_emotion_set_is_supported()
        self.verbose = verbose
        self.target_dimensions = (48, 48)
        self.channels = 1
        self._initialize_model()

    def _initialize_model(self):
        logger.info('Initializing FER model parameters for target emotions: %s' % self.target_emotions)
        self.model, self.emotion_map = self._choose_model_from_target_emotions()

    def predict(self, image):
        """
        Predicts discrete emotion for given image.

        :param images: image file (jpg or png format)
        """
        # image = misc.imread(image_file)
        gray_image = image
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, self.target_dimensions, interpolation=cv2.INTER_LINEAR)
        final_image = np.array([np.array([resized_image]).reshape(list(self.target_dimensions)+[self.channels])])
        prediction = self.model.predict(final_image)
        # self._print_prediction(prediction[0])
        return prediction[0]

    def _check_emotion_set_is_supported(self):
        """
        Validates set of user-supplied target emotions.
        """
        supported_emotion_subsets = [
            set(['anger', 'fear', 'surprise', 'calm']),
            set(['happiness', 'disgust', 'surprise']),
            set(['anger', 'fear', 'surprise']),
            set(['anger', 'fear', 'calm']),
            set(['anger', 'happiness', 'calm']),
            set(['anger', 'fear', 'disgust']),
            set(['calm', 'disgust', 'surprise']),
            set(['sadness', 'disgust', 'surprise']),
            set(['anger', 'happiness'])
        ]
        if not set(self.target_emotions) in supported_emotion_subsets:
            error_string = 'Target emotions must be a supported subset. '
            error_string += 'Choose from one of the following emotion subset: \n'
            possible_subset_string = ''
            for emotion_set in supported_emotion_subsets:
                possible_subset_string += ', '.join(emotion_set)
                possible_subset_string += '\n'
            error_string += possible_subset_string
            raise ValueError(error_string)

    def _choose_model_from_target_emotions(self):
        """
        Initializes pre-trained deep learning model for the set of target emotions supplied by user.
        """
        model_indices = [self.emotion_index_map[emotion] for emotion in self.target_emotions]
        sorted_indices = [str(idx) for idx in sorted(model_indices)]
        model_suffix = ''.join(sorted_indices)
        model_file = 'models/conv_model_%s.hdf5' % model_suffix
        emotion_map_file = 'models/conv_emotion_map_%s.json' % model_suffix
        emotion_map = json.loads(open(resource_filename('EmoPy',emotion_map_file)).read())
        return load_model(resource_filename('EmoPy',model_file)), emotion_map
