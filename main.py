import argparse
import glob
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import yaml
from multimodal.utils import timer
from prettytable import PrettyTable

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__ if __name__ != '__main__' else 'main')


def main():
    """Define possible options for the CLI."""

    parser = argparse.ArgumentParser(description='Sentiment recognition in Spanish-language videos. Three modalities could ' +
    'be considered: text, audio and video. The analysis could be accomplished taking into account one, two or the three modalities.')

    parser.add_argument('--file', help='Configuration file with the analysis that is going to be performed')

    parser.add_argument('--bounds', help='Bounds file needed for synchronization', nargs=1)
    parser.add_argument('--text', help='Text file for the analysis in CSV format', nargs=1)
    parser.add_argument('--audio', help='Audio files for the analysis in WAV format', nargs='+')
    parser.add_argument('--audio-dir', help='Directory with the audio files in WAV format', nargs=1)
    parser.add_argument('--video', help='Video file for the analysis in MP4 format', nargs=1)

    args = parser.parse_args()

    # Error management when using command line

    if args.file is not None and list(vars(args).values()).count(None) != (len(vars(args)) - 1):
        parser.error('If --file option is used, any other option cannot be used')

    # Load parameters from configuration file

    if args.file is not None:
        logger.info('Loading configuration file...')
        configuration = read_config_file(args.file)
        args.bounds = configuration['bounds'] if 'bounds' in configuration else None
        args.text = configuration['text'] if 'text' in configuration else None
        args.audio = configuration['audio'] if 'audio' in configuration else None
        args.audio_dir = configuration['audio_dir'] if 'audio_dir' in configuration else None
        args.video = configuration['video'] if 'video' in configuration else None

    # Error management between options

    if args.audio is not None and args.audio_dir is not None:
        parser.error('Cannot use --audio and --audio-dir at the same time')
    elif (list(vars(args).values()).count(None) != (len(vars(args)) - 1) and
    (args.audio is not None or args.text is not None) and args.bounds is None):
        parser.error('A bounds file is needed for perform an analysis')

    # Fileformat error management

    if args.text is not None and os.path.splitext(args.text[0])[1].lower() != '.csv':
        parser.error('File format {} for --text not supported'.format(os.path.splitext(args.text[0])[1]))
    if args.video is not None and os.path.splitext(args.video[0])[1].lower() != '.mp4':
        parser.error('File format {} for --video not supported'.format(os.path.splitext(args.video[0])[1]))
    for audio in args.audio or []:
        if args.audio is not None and os.path.splitext(audio)[1].lower() != '.wav':
            parser.error('File format {} for --audio not supported'.format(os.path.splitext(audio)[1]))

    # Run analysis
    from multimodal import analyzer
    from multimodal import features
    results = {}

    if args.bounds is not None:
        args.bounds = pd.read_csv(args.bounds[0], sep=';')

    if args.audio_dir is not None:
        args.audio = sorted(glob.glob(os.path.join(args.audio_dir[0], '*.wav')))

    if args.text is not None:
        # Text analysis
        logger.info('Computing analysis for text...')
        with timer('Text analysis', logger.info):
            args.text = pd.read_csv(args.text[0], sep=';')
            results['Text'] = [analyzer.text_analyzer(el) for el in args.text.transcription]

    if args.audio is not None:
        # Audio analysis
        logger.info('Computing analysis for audio...')
        args.audio = sorted(args.audio)
        results['Audio'] = []
        with timer('Audio analysis', logger.info):
            for file in args.audio:
                ft = features.AudioFeatures().run(file)
                results['Audio'].append(analyzer.audio_analyzer(ft))

    if args.video is not None and args.bounds is None:
        # Video analysis of each frame
        logger.info('Computing analysis for video...')
        results['Video'] = []
        with timer('Video analysis', logger.info):
            ft = features.VideoFeatures().run(args.video[0])
            for index, row in ft.iterrows():
                del row['timestamp']
                results['Video'].append(analyzer.video_analyzer(row))

    if args.video is not None and args.bounds is not None:
        # Video analysis synchronized with the other sources
        logger.info('Computing analysis for video...')
        results['Video'] = []
        with timer('Video analysis', logger.info):
            ft = features.BimodalFeatures(args.bounds).run_video(args.video[0]).video
            for index, row in ft.iterrows():
                del row['start']
                del row['end']
                results['Video'].append(analyzer.video_analyzer(row))

    if args.text is not None and args.audio is not None:
        # Text and audio analysis
        logger.info('Computing analysis using two modalities: text + audio...')
        with timer('Text + Audio analysis', logger.info):
            mods = ('text', 'audio')
            ft = features.BimodalFeatures(args.bounds).run(mods, text=args.text, audio=args.audio)
            results['Text + Audio'] = analyzer.bimodal_analyzer(ft, mods)

    if args.audio is not None and args.video is not None:
        # Audio and video analysis
        logger.info('Computing analysis using two modalities: audio + video...')
        with timer('Audio + Video analysis', logger.info):
            mods = ('audio', 'video')
            ft = features.BimodalFeatures(args.bounds).run(mods, audio=args.audio, video=args.video[0])
            results['Audio + Video'] = analyzer.bimodal_analyzer(ft, mods)

    if args.video is not None and args.text is not None:
        # Video and text analysis
        logger.info('Computing analysis using two modalities: video + text...')
        with timer('Video + Text analysis', logger.info):
            mods = ('video', 'text')
            ft = features.BimodalFeatures(args.bounds).run(mods, video=args.video[0], text=args.text)
            results['Video + Text'] = analyzer.bimodal_analyzer(ft, mods)

    if args.text is not None and args.audio is not None and args.video is not None:
        # Multimodal analysis
        logger.info('Computing analysis using three modalities: text + audio + video...')
        with timer('Multimodal analysis', logger.info):
            ft = features.MultimodalFeatures(args.bounds).run(text=args.text, audio=args.audio, video=args.video[0])
            results['Multimodal'] = analyzer.multimodal(ft)

    # Show results
    if args.video is not None and args.bounds is None:
        plt.plot(results['Video'])
        plt.ylabel('Sentiment')
        plt.xlabel('Time')
        plt.show()
    else:
        show_results(results, args.bounds)


def read_config_file(file):
    """Read configuration file"""
    with open(file, 'r') as stream:
        return yaml.safe_load(stream)


def show_results(results, bounds):
    table = PrettyTable()
    table.field_names = ['Modality'] + ['%0.3f' % (end) for end in bounds['#endtime']] + ['Majority']
    table.align['Modality'] = 'l'
    table.align['Majority'] = 'r'

    for end in bounds['#endtime']:
        name = '%0.3f' % (end)
        table.align[name] = 'r'

    for k, v in dict.items(results):
        v = [ int(x) for x in v ]
        table.add_row([k] + v + [max(v, key=v.count)])

    print()
    print(table.get_string(title="Results from multimodal analysis"))
    print()


if __name__ == "__main__":
    main()
