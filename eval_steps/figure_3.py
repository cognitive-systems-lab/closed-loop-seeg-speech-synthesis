import sys
sys.path.append('..')
import argparse
import configparser
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')
from matplotlib import rc
from local.offline import pearson_correlation, extract_corrs_for_distribution
import glob
import h5py
from local.data_loader import Session
import logging
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, mannwhitneyu

# Script used in final version

logger = logging.getLogger('figure_3.py')


# ----- Font settings

font = {'family': 'sans-serif', 'sans-serif': ['Helvetica']}
rc('font',**font)
rc('text', usetex=True)
rc('axes', labelsize='large')
rc('xtick', labelsize='large')
rc('ytick', labelsize='medium')
rc('legend', fontsize='large')


def plot_figure_3(session_dir, dest_dir):
    exp_dir = os.path.join(dest_dir, 'exp1')

    orig = np.load(os.path.join(exp_dir, 'orig.npy'))
    reco = np.load(os.path.join(exp_dir, 'pm_reco.npy'))

    pairs = [(orig[i:i+200], reco[i:i+200]) for i in range(0, len(orig), 300)]
    pair_corrs = [pearson_correlation(o, r)[0] for o, r in pairs]
    top_five_idx = np.argsort(pair_corrs)[::-1][:5]

    cherry_picked_orig = np.vstack([pairs[i][0] for i in top_five_idx])
    cherry_picked_reco = np.vstack([pairs[i][1] for i in top_five_idx])

    sess = Session(session_dir)
    words = [sess.words[i] for i in top_five_idx]

    logger.info('Top five words: {}'.format(words))

    # Define Figure 4
    fig = plt.figure(figsize=(14, 6))

    ax_o = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    ax_r = plt.subplot2grid((2, 3), (1, 0), colspan=2)
    ax_c = plt.subplot2grid((2, 3), (0, 2), rowspan=2)

    # -------- Draw selected spectrograms -------------- #
    nb_xticks = 5

    ax_o.set_ylabel('Mel-Freq. [in kHz]')
    o_title = ax_o.set_title('Original')
    o_title.set_position([0.5, 1.18])
    ax_o.imshow(cherry_picked_orig.T, aspect='auto', origin='lower')
    ax_o.grid(False)
    ax_o.xaxis.tick_top()
    ax_o.set_xticks([100, 300, 500, 700, 900])
    ax_o.set_xticklabels(words)

    for i in range(1, 5):
        line_x = i * 200
        ax_o.axvline(line_x, 0, 39, color='white', alpha=1, linewidth=2, linestyle='--')

    ax_o.set_yticks(np.linspace(0, 39, nb_xticks))
    ax_o.set_ylim(0, 39)
    ax_o.set_yticklabels(np.linspace(0, 8, nb_xticks, dtype=int))

    props = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 12, 'shrinkB': 12, 'linewidth': 2, 'color': 'white'}
    ax_o.annotate('1 Second', xy=(60, 32), zorder=10, annotation_clip=False, color='white')
    ax_o.annotate('', xy=(50, 26), xytext=(150, 26), arrowprops=props, annotation_clip=False)

    ax_r.set_ylabel('Mel-Freq. [in kHz]')
    r_title = ax_r.set_title('Reconstruction')
    r_title.set_position([0.5, 1.05])
    ax_r.imshow(cherry_picked_reco.T, aspect='auto', origin='lower')
    ax_r.grid(False)

    ax_r.set_xticks([])
    for i in range(1, 5):
        line_x = i * 200
        ax_r.axvline(line_x, 0, 39, color='white', alpha=1, linewidth=2, linestyle='--')

    ax_r.set_yticks(np.linspace(0, 39, nb_xticks))
    ax_r.set_ylim(0, 39)
    ax_r.set_yticklabels(np.linspace(0, 8, nb_xticks, dtype=int))

    # -------- Draw Correlation results -------------- #
    # pm_dist_means, pm_dist_stds = extract_corrs_for_distribution(orig, reco)

    n_folds = 10
    kf = KFold(n_splits=n_folds)

    rs = np.zeros((n_folds, orig.shape[1]))
    for k, (train, test) in enumerate(kf.split(orig)):
        o = orig[test, :]
        r = reco[test, :]

        for spec_bin in range(o.shape[1]):
            c = pearsonr(o[:, spec_bin], r[:, spec_bin])[0]
            rs[k, spec_bin] = c

    pm_dist_means = np.mean(rs, axis=0)
    pm_dist_stds = np.std(rs, axis=0)
    rs_for_significance_test = rs

    rc_corrs = []
    for i in range(1, 101):
        rc_reco = np.load(os.path.join(exp_dir, 'rc_reco_i={:03}.npy'.format(i)))

        n_folds = 10
        kf = KFold(n_splits=n_folds)

        rs = np.zeros((n_folds, orig.shape[1]))
        for k, (train, test) in enumerate(kf.split(orig)):
            o = orig[test, :]
            r = rc_reco[test, :]

            for spec_bin in range(o.shape[1]):
                c = pearsonr(o[:, spec_bin], r[:, spec_bin])[0]
                rs[k, spec_bin] = c

        rc_corrs.append(rs)

    # rc_corrs = [pearson_correlation(orig, np.load(f), return_means=True)[2] for f in glob.glob(os.path.join(exp_dir, 'rc_reco_i=*.npy'))]
    rc_corrs = np.vstack(rc_corrs)
    rc_dist_means = np.mean(rc_corrs, axis=0)
    rc_dist_stds = np.std(rc_corrs, axis=0)

    for spec_bin in range(40):
        stat, p = mannwhitneyu(rs_for_significance_test[:, spec_bin], rc_corrs[:, spec_bin])
        logger.info('Spec Bin: {}, Stat: {}, p: {}, p (Bonferoni): {}'.format(spec_bin, stat, p, p*40))

    ax_c.grid(False)
    ax_c.yaxis.grid(True, color='lightgrey', linestyle='dashed')
    ax_c.set_axisbelow(True)
    ax_c.spines['bottom'].set_visible(True)
    ax_c.spines['bottom'].set_color('lightgrey')
    ax_c.spines['bottom'].set_linestyle(ax_c.yaxis.get_gridlines()[0].get_linestyle())
    ax_c.spines['bottom'].set_linewidth(ax_c.yaxis.get_gridlines()[0].get_linewidth())

    ax_c.set_facecolor('white')
    ax_c.plot(pm_dist_means, c='b')
    ax_c.fill_between(np.arange(len(pm_dist_means)), pm_dist_means - pm_dist_stds, pm_dist_means + pm_dist_stds,
                      facecolor='dodgerblue', alpha=0.5)

    ax_c.plot(rc_dist_means, c='r')
    ax_c.fill_between(np.arange(len(rc_dist_means)), rc_dist_means - rc_dist_stds, rc_dist_means + rc_dist_stds,
                      facecolor='salmon', alpha=0.5)

    nb_xticks = 5

    ax_c.set_ylabel('Pearson Correlation')
    ax_c.set_ylim(-0.1, 0.85)
    ax_c.set_yticks(np.arange(-0.1, 0.9, 0.1))
    ax_c.set_yticklabels(['{:.01f}'.format(l) for l in np.arange(-0.1, 0.9, 0.1)])
    ax_c.set_xlabel('Hz')
    ax_c.set_xticks(np.linspace(0, 39, nb_xticks))
    ax_c.set_xlim(0, 39)
    ax_c.set_xticklabels(np.linspace(0, 8000, nb_xticks, dtype=int))
    leg = ax_c.legend(['Proposed method', 'Chance level'], loc=1, ncol=2, facecolor='white', framealpha=0,
                      bbox_to_anchor=(1, 1.1))
    leg.get_frame().set_edgecolor('black')

    ax_o.annotate(r'\textbf{a}', xy=(-60, 46), zorder=10, annotation_clip=False, fontsize=16, weight='bold')
    ax_c.annotate(r'\textbf{b}', xy=(-8, 0.924), zorder=10, annotation_clip=False, fontsize=16, weight='bold')

    plt.subplots_adjust(left=0.06, bottom=0.08, right=0.97, top=0.86, wspace=0.3, hspace=0.3)
    plt.savefig(os.path.join(dest_dir, 'figure_3.png'), dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Create Figure 3.')
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

    plot_figure_3(session_dir=session_dir, dest_dir=dest_dir)
    logger.info('Saved figure 3 in {}.'.format(dest_dir))
