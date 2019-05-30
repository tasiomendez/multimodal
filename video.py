import cv2
import logging
import numpy as np
import os
import pandas as pd
from .fermodel import FERModel
from .utils import progress_bar

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageAnalyzer:

    def __init__(self, classifier, verbose=True, **kwargs):
        self.verbose = verbose
        self.classifier = classifier

    def load(self, path):
        if os.path.isfile(path):
            return cv2.imread(path)
        else:
            raise Exception('File do not exists!')

    def toGray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def toRGB(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def crop(self, image, coordinates, x_off=20, y_off=40):
        x, y, width, height = coordinates
        x1, x2 = (x - x_off, x + width  + x_off)
        y1, y2 = (y - y_off, y + height + y_off)
        return image[y1:y2, x1:x2]

    def faces(self, image, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48)):
        cascade = cv2.CascadeClassifier(self.classifier)
        img_equalized = cv2.equalizeHist(image)

        return cascade.detectMultiScale(img_equalized,
            scaleFactor = scaleFactor, minNeighbors = minNeighbors,
            minSize = minSize, flags = cv2.CASCADE_SCALE_IMAGE)

    def drawBox(self, image, color=(0, 255, 255)):
        _color = np.asarray(color).astype(int).tolist()
        gray = self.toGray(image)
        _image = image.copy()
        for face in self.faces(gray):
            x, y, width, height = face
            cv2.rectangle(_image, (x, y), (x + width, y + height), _color, 2)
        return _image


class VideoAnalyzer(ImageAnalyzer):

    def __init__(self, classifier, file, emotions=['anger', 'happiness', 'calm'], verbose=True, **kwargs):
        if os.path.isfile(file) and os.path.splitext(file)[1] == '.mp4':
            video = cv2.VideoCapture(file)
            self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = round(video.get(cv2.CAP_PROP_FPS))
            self.nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.target = video
        else:
            raise Exception('File format do not supported!')

        self.emotions = emotions
        self.model = FERModel(self.emotions, verbose=verbose)
        self.verbose = verbose
        super().__init__(classifier, verbose=verbose, **kwargs)

    def frames(self):
        """Frames generator"""
        success, frame = self.target.read()
        while success:
            yield frame
            success, frame = self.target.read()

    def analyze(self, **kwargs):
        self.results = []
        for frame in self.frames():
            if (self.verbose):
                progress_bar(len(self.results), self.nframes, prefix='Progress: ', bar_length=80)
            gray = self.toGray(frame)
            faces = self.faces(gray)
            # Get face if it is detected else whole frame
            face = faces[0] if len(faces) > 0 else (0, 0, frame.shape[1], frame.shape[0])
            prediction = self.model.predict(self.crop(frame, face, x_off=0, y_off=0))
            self.results.append({
                'frame': frame,
                'emotions': { emotion:  value/sum(prediction) for emotion, value in zip(self.emotions, prediction) },
                'timestamp': len(self.results) / self.fps
            })
        return self

    def toDataFrame(self):
        _images = self.results.copy()
        for i in range(len(_images)):
            _images[i] = dict(_images[i], **_images[i]['emotions'])
            del _images[i]['emotions']
            del _images[i]['frame']
        return pd.DataFrame(_images)
