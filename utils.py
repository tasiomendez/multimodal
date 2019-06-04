import logging
import re
import string
import sys
from contextlib import contextmanager
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from time import time

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def custom_tokenizer(words):
    """Preprocessing tokens as seen in the lexical notebook"""
    tokenizer = ToktokTokenizer() # Spanish Tokenizer
    tokens = tokenizer.tokenize(words.lower())
    snow = SnowballStemmer('spanish')
    stems = [snow.stem(t) for t in tokens]
    stoplist = stopwords.words('spanish')
    stems_clean = [w for w in stems if w not in stoplist]
    punctuation = set(string.punctuation + '¡¿')
    stems_punct = [w for w in stems_clean if  w not in punctuation]
    stems_pause = [w for w in stems_punct if not re.search('^(-)|(-)$', w)]
    stems_clean = [re.sub('/', '', w) for w in stems_pause]

    return stems_clean


def progress_bar(iteration, total, prefix='', suffix='', decimals=2, bar_length=100):
    str_format = "{0:." + str(decimals) + "f}"
    if iteration == total-1:
        percents = str_format.format(float(100))
    else:
        percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '=' * filled_length + '>' + '-' * (bar_length - filled_length - 1)

    sys.stdout.write('\r%s [%s] %s %s %s' % (prefix, bar, percents, '%', suffix))

    if iteration == total-1:
        sys.stdout.write('\n')
    sys.stdout.flush()


@contextmanager
def timer(name='task', function=logger.info):
    """Auxiliar function as timer"""
    start = time()
    yield start
    end = time()
    function('{} in {} seconds'.format(name, end - start))
