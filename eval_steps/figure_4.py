import sys
sys.path.append('..')
import argparse
import configparser
from local.data_loader import DecodingRun
from eval_steps.exp3 import Experiment3
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')
from matplotlib import rc
from scipy.stats import mannwhitneyu
import logging


logger = logging.getLogger('figure_4.py')


# Global adjustments regarding Matplotlib
font = {'family': 'sans-serif', 'sans-serif': ['Helvetica']}
rc('font',**font)
rc('text', usetex=True)
rc('axes', labelsize='large')
rc('xtick', labelsize='large')
rc('ytick', labelsize='medium')
rc('legend', fontsize='large')


def plot_figure_4(session_dir, dest_dir):

    # Load experiment data
    exp2_dir = os.path.join(dest_dir, 'exp2')
    chance_imagine = np.load(os.path.join(exp2_dir, 'exp2_imagine_chance.npy'))
    chance_whisper = np.load(os.path.join(exp2_dir, 'exp2_whisper_chance.npy'))

    decoding_imagine = np.load(os.path.join(exp2_dir, 'exp2_imagine_pm.npy'))
    decoding_whisper = np.load(os.path.join(exp2_dir, 'exp2_whisper_pm.npy'))

    nb_nan = np.count_nonzero(np.isnan(chance_imagine))
    if nb_nan > 0:
        chance_imagine = chance_imagine[~np.isnan(chance_imagine)]

    nb_nan = np.count_nonzero(np.isnan(chance_whisper))
    if nb_nan > 0:
        chance_whisper = chance_whisper[~np.isnan(chance_whisper)]

    logger.info('Median DTW scores (whisper) {} + {})'.format(np.median(decoding_whisper), np.std(decoding_whisper)))
    logger.info('Median DTW scores (imagine) {} + {})'.format(np.median(decoding_imagine), np.std(decoding_imagine)))

    logger.info('Chance DTW scores (whisper) {} + {})'.format(np.median(chance_whisper), np.std(chance_whisper)))
    logger.info('Chance DTW scores (imagine) {} + {})'.format(np.median(chance_imagine), np.std(chance_imagine)))

    logger.info('Number of overlapping words with audible run regarding whisper: {}'.format(len(decoding_whisper)))
    logger.info('Number of overlapping words with audible run regarding imagine: {}'.format(len(decoding_imagine)))

    logger.info('Mann-Whitney U Test imagine: {}'.format(mannwhitneyu(decoding_imagine, chance_imagine)))
    logger.info('Mann-Whitney U Test whisper: {}'.format(mannwhitneyu(decoding_whisper, chance_whisper)))
    logger.info('Mann-Whitney U whisper vs. imagine: {}'.format(mannwhitneyu(decoding_whisper, decoding_imagine)))

    # Define Figure 4
    fig = plt.figure(figsize=(12, 6.5))

    ax_w = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax_i = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    ax_a = plt.subplot2grid((2, 3), (1, 2))
    ax_b = plt.subplot2grid((2, 3), (0, 2))

    # ----- Draw selected examples ------ #

    words_whisper = ['maantje', 'sok', 'meisjes', 'tak', 'sprong']
    words_imagine = ['groen', 'vloog', 'geen', 'zonlicht', 'zou']

    whisper_run = DecodingRun(os.path.join(session_dir, 'whisper'))
    imagine_run = DecodingRun(os.path.join(session_dir, 'imagine'))

    whisper_audios = [whisper_run.get_trial_by_word(word)[2] for word in words_whisper]
    imagine_audios = [imagine_run.get_trial_by_word(word)[2] for word in words_imagine]

    # Trial based normalization
    whisper_audios = [trial / 2**15 for trial in whisper_audios]
    imagine_audios = [trial / 2**15 for trial in imagine_audios]

    whisper_audios = np.hstack(whisper_audios)
    imagine_audios = np.hstack(imagine_audios)

    ax_w.plot(whisper_audios, color='steelblue')
    ax_w.set_facecolor('white')
    ax_w.set_yticks([])
    ax_w.set_xlim(0, len(whisper_audios))
    ax_w.set_xticks(np.array([0.5, 1.5, 2.5, 3.5, 4.5]) * 32000)
    ax_w.set_xticklabels(words_whisper)
    ax_w.xaxis.tick_top()
    ax_w.set_ylabel('Amplitude')
    ax_w.set_ylim(-1, 1)
    ttl_w = ax_w.set_title('Whisper')
    ttl_w.set_position([.5, 1.2])

    ax_w.spines['top'].set_visible(False)
    ax_w.spines['right'].set_visible(False)
    ax_w.spines['left'].set_visible(False)
    ax_w.spines['bottom'].set_visible(False)

    for i in range(1, len(words_whisper)):
        line_x = i * 32000
        ax_w.axvline(line_x, -10000, 1000, color='#505050', alpha=1, linewidth=2, linestyle='--')

    ax_i.plot(imagine_audios, color='steelblue')
    ax_i.set_facecolor('white')
    ax_i.set_yticks([])
    ax_i.set_xlim(0, len(imagine_audios))
    ax_i.set_xticks(np.array([0.5, 1.5, 2.5, 3.5, 4.5]) * 32000)
    ax_i.set_xticklabels(words_imagine)
    ax_i.xaxis.tick_top()
    ax_i.set_ylabel('Amplitude')
    ax_i.set_ylim(-1, 1)
    ttl = ax_i.set_title('Imagine')
    ttl.set_position([0.5, 1.2])

    ax_i.spines['top'].set_visible(False)
    ax_i.spines['right'].set_visible(False)
    ax_i.spines['left'].set_visible(False)
    ax_i.spines['bottom'].set_visible(False)

    for i in range(1, len(words_imagine)):
        line_x = i * 32000
        ax_i.axvline(line_x, -10000, 1000, color='#505050', alpha=1, linewidth=2, linestyle='--')

    props = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 8, 'shrinkB': 8, 'linewidth': 2, 'color': '#505050'}
    ax_i.annotate('1 Second', xy=(8000, -1.25), zorder=10, annotation_clip=False)
    ax_i.annotate('', xy=(24000, -0.8), xytext=(8000, -0.8), arrowprops=props)

    # Draw boxplots
    data_pm = [decoding_whisper, decoding_imagine]
    data_ch = [chance_whisper, chance_imagine]

    bplot_pm = ax_b.boxplot(data_pm, positions=[1, 3], patch_artist=True, widths=0.4)
    bplot_ch = ax_b.boxplot(data_ch, positions=[2, 4], patch_artist=True, widths=0.4)
    ax_b.set_ylabel('DTW Correlation Coefficient')
    ax_b.set_xticks([1.5, 3.5])
    ax_b.set_xticklabels(['Whisper', 'Imagine'])
    ax_b.grid(False)
    ax_b.set_facecolor('white')
    ax_b.set_xlim(0.5, 4.5)

    bplot_pm['medians'][0].set_color('black')
    bplot_pm['medians'][0].set_linewidth(2)

    bplot_ch['medians'][0].set_color('black')
    bplot_ch['medians'][0].set_linewidth(2)

    bplot_pm['medians'][1].set_color('black')
    bplot_pm['medians'][1].set_linewidth(2)

    bplot_ch['medians'][1].set_color('black')
    bplot_ch['medians'][1].set_linewidth(2)

    bplot_pm['boxes'][0].set_facecolor('lightblue')
    bplot_ch['boxes'][0].set_facecolor('plum')

    bplot_pm['boxes'][1].set_facecolor('lightblue')
    bplot_ch['boxes'][1].set_facecolor('plum')

    leg = ax_b.legend([bplot_pm['boxes'][0], bplot_ch['boxes'][0]], ['Proposed method', 'Chance'],
                      loc=1, ncol=2, facecolor='white', framealpha=0, bbox_to_anchor=(1, 1.3))

    props = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 10, 'shrinkB': 10, 'linewidth': 2, 'color': '#505050'}
    ax_b.annotate('***', xy=(1.3, 1), zorder=10, annotation_clip=False, fontsize=16)
    ax_b.annotate('', xy=(1, 0.85), xytext=(2, 0.85), arrowprops=props, annotation_clip=False)

    ax_b.annotate('**', xy=(3.4, 1.0), zorder=10, annotation_clip=False, fontsize=16)
    ax_b.annotate('', xy=(3, 0.85), xytext=(4, 0.85), arrowprops=props, annotation_clip=False)

    y_labels = []
    for u in ax_b.get_yticklabels():
        t = u.get_text()
        t = t.replace('$', '')
        y_labels.append(t)

    ax_b.set_ylim(-0.3, 0.95)
    ax_b.set_yticks(np.arange(-0.2, 0.9, 0.2))
    ax_b.set_yticklabels([round(v, ndigits=1) for v in np.arange(-0.2, 0.9, 0.2)])

    # ----- Draw amount of speech ------ #

    results = []
    for run in ['whisper', 'imagine']:
        run_dir = os.path.join(session_dir, run)
        exp3 = Experiment3(config, run_dir)
        amount_of_speech_during_trials, amount_of_speech_during_rest = exp3.run()

        results.append((run, amount_of_speech_during_trials, amount_of_speech_during_rest))
        print(run, amount_of_speech_during_trials, amount_of_speech_during_rest)

    labels, speech_in_trials, speech_outside_trials = zip(*results)
    labels = [label.title() for label in labels]

    logger.info('Normalizing amount of speech during trials')

    # Calculate percentage of decoded speech during trials and resting phases. The experiment consists of 100 trials
    # each 2 seconds and 100 resting phases each 1 second.
    speech_in_trials = [v / 200 for v in speech_in_trials]
    speech_outside_trials = [v / 100 for v in speech_outside_trials]

    x = np.arange(len(labels))
    width = 0.35

    ax_a.bar(x - width/2, speech_in_trials, width, color='dodgerblue', label='During trials')
    ax_a.bar(x + width/2, speech_outside_trials, width, color='salmon', label='Outside trials')

    ax_a.set_facecolor('white')
    ax_a.set_ylabel('Proportion of Decoded Speech', labelpad=10)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_yticklabels(['0.0', '0.1', '0.2', '0.3'])
    ax_a.legend(loc=1, ncol=1, facecolor='white', framealpha=0, bbox_to_anchor=[1, 1.1])

    ax_w.annotate(r'\textbf{a}', xy=(-4000, 1.5), zorder=10, annotation_clip=False, fontsize=16, weight='bold')
    ax_i.annotate(r'\textbf{b}', xy=(-4000, 1.5), zorder=10, annotation_clip=False, fontsize=16, weight='bold')
    ax_b.annotate(r'\textbf{c}', xy=(-0.5, 1.27), zorder=10, annotation_clip=False, fontsize=16, weight='bold')
    ax_a.annotate(r'\textbf{d}', xy=(-0.905, 0.45), zorder=10, annotation_clip=False, fontsize=16, weight='bold')

    plt.subplots_adjust(left=0.03, bottom=0.07, right=0.98, top=0.85, wspace=0.3, hspace=0.4)

    filename = os.path.join(dest_dir, 'figure_4.png')
    plt.savefig(filename, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Create Figure 4.')
    parser.add_argument('config', help='Path to experiment config file.')

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    # initialize logging handler
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)])

    session_dir = os.path.join(config['General']['storage_dir'], config['General']['session'])
    dest_dir = os.path.join(config['General']['temp_dir'], config['General']['session'])

    logger.info('Session dir: {}'.format(session_dir))
    logger.info('Dest dir: {}'.format(dest_dir))

    plot_figure_4(session_dir=session_dir, dest_dir=dest_dir)
    logger.info('Saved figure 4 in {}.'.format(dest_dir))
