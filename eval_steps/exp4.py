import pickle
import argparse
import configparser
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import matplotlib
from matplotlib import rc
import re
import sys
import logging
import matplotlib.patches as patches
matplotlib.style.use('ggplot')

logger = logging.getLogger('exp4.py')

# global matplotlib style configurations (for being consistent across plots)
font = {'family': 'sans-serif', 'sans-serif': ['Helvetica']}
rc('font', **font)
rc('text', usetex=True)
rc('axes', labelsize='large')
rc('xtick', labelsize='large')
rc('ytick', labelsize='medium')
rc('legend', fontsize='large')

plt.rcParams['xtick.labelsize'] = 2


class Experiment4:
    def __init__(self, session_dir):
        self.session_dir = session_dir
        self.channel_names = ['LB1', 'LB2', 'LB3', 'LB4', 'LB5', 'LB6', 'LB7', 'LB8', 'LC1', 'LC2', 'LC3', 'LC4', 'LC5',
                              'LC6', 'LC7', 'LC8', 'LC9', 'LC10', 'LC11', 'LC12', 'LD1', 'LD2', 'LD3', 'LD4', 'LD5',
                              'LD6', 'LD7', 'LD8', 'LE1', 'LE2', 'LE3', 'LE4', 'LE5', 'LE6', 'LE7', 'LE8', 'LE9',
                              'LE10', 'LF1', 'LF2', 'LF3', 'LF4', 'LF5', 'LG1', 'LG2', 'LG3', 'LG4', 'LG5', 'LG6',
                              'LG7', 'LG8', 'LG9', 'LG10', 'LG11', 'LG12', 'LG13', 'LG14', 'LG15', 'LG16', 'LG17',
                              'LG18', 'LI1', 'LI2', 'LI3', 'LI4', 'LI5', 'LI6', 'LI7', 'LI8', 'LI9', 'LI10', 'LI11',
                              'LI12', 'LI13', 'LI14', 'LI15', 'LI16', 'LI17', 'LI18', 'LK1', 'LK2', 'LK3', 'LK4', 'LK5',
                              'LK6', 'LK7', 'LK8', 'LK9', 'LK10', 'LM1', 'LM2', 'LM3', 'LM4', 'LM5', 'LM6', 'LM7',
                              'LM8', 'LT1', 'LT2', 'LT3', 'LT4', 'LT5', 'LT6', 'LT8', 'LT9', 'LT10', 'LU2', 'LU3',
                              'LU4', 'LU5', 'LU6', 'LU7', 'LU8', 'LU9', 'LU10', 'LU11', 'LU12', 'E124', 'E125', 'E126',
                              'E127', 'E128']

        lda_path = os.path.join(self.session_dir, 'params.h5')  # This information should be in the Session
        with h5py.File(lda_path) as hf:
            self.estimators = pickle.loads(hf['estimators'][...].tobytes())
            self.select = hf['select'][:]

        channels = ['{}-{}'.format(n, t) for n in self.channel_names for t in reversed(range(0, 5))]
        self.sel_features = [f for i, f in enumerate(channels) if i in self.select]

        self.obs_data = np.load(os.path.join(session_dir, 'training_features.npy'))

        # Colormap for the electrode shafts
        self.tab10_modified = (
            (1.0, 0.4980392156862745, 0.054901960784313725),  # LB
            (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # LC
            (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # LD
            (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),  # LE
            (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),  # LF
            (0.99215686274509807, 0.75294117647058822, 0.52549019607843139),  # LG
            (0.65098039215686276, 0.80784313725490198, 0.8901960784313725),  # LI
            (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),  # LK
            (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),  # LM
            (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # LT
            (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # LU
        )

    def compute_activations(self):
        # Calculate W
        w = np.zeros((150, 9, 40))
        for i in range(len(self.estimators)):
            est = self.estimators[i]

            mask = np.ones(9, dtype=bool)
            if i == 7:
                mask[1] = False

            if i == 14:
                mask[1] = False

            w[:, mask, i] = est.coef_.T

        # Compute all Activations
        sigma = np.cov(self.obs_data.T)
        all_A = np.zeros((len(self.select), 9, len(self.estimators)))

        for i in range(len(self.estimators)):
            W = w[:, :, i]
            sn = np.array([W.T @ self.obs_data[x_n, :] for x_n in range(len(self.obs_data))])
            sigma_s = np.cov(sn.T)
            try:
                if i == 7 or i == 14:
                    mask = np.ones(9, dtype=bool)
                    mask[1] = False
                    sigma_s_inv = np.linalg.inv(sigma_s[mask, :][:, mask])  # Add missing column
                    tmp = np.zeros((9, 9))
                    tmp[mask, :][:, mask] = sigma_s_inv
                    sigma_s_inv = tmp
                else:
                    sigma_s_inv = np.linalg.inv(sigma_s)

                A = sigma @ W @ sigma_s_inv
                all_A[:, :, i] = A
            except np.linalg.LinAlgError:
                print('Singualar value error in index: {}'.format(i))

        # Calculate average activations and assign only values based on the feature selection
        activations = np.mean(np.abs(all_A), axis=(1, 2))

        matrix = np.zeros((len(self.channel_names), 5))
        for f in self.sel_features:
            b, t = f.split('-')
            matrix[self.channel_names.index(b), int(t)] = activations[self.sel_features.index(f)]

        return matrix

    def plot_results(self, activations, filename):
        electrode_channel_numbering = []
        for ch in self.channel_names:
            m = re.match(r"([a-z]+)([0-9]+)", ch, re.I)
            n = m.groups()[1]
            electrode_channel_numbering.append(int(n))

        minimum = np.min(activations)
        maximum = np.max(activations)

        vmax = np.max([abs(minimum), maximum])

        # Feature selection polygon (manual)
        polygon = [(-0.5, 4.5), (0.5, 4.5), (0.5, 3.5), (1.5, 3.5), (1.5, 2.5), (2.5, 2.5), (2.5, 2.5), (3.5, 2.5),
                   (3.5, 1.5), (4.5, 1.5), (4.5, 0.5), (5.5, 0.5), (5.5, -0.5), (6.5, -0.5), (6.5, -0.5), (7.5, -0.5),
                   (7.5, 3.5), (8.5, 3.5), (8.5, 3.5), (9.5, 3.5), (9.5, 3.5), (10.5, 3.5), (10.5, 3.5), (11.5, 3.5),
                   (11.5, 3.5), (12.5, 3.5), (12.5, 3.5), (13.5, 3.5), (13.5, 3.5), (14.5, 3.5), (14.5, 3.5),
                   (15.5, 3.5), (15.5, 3.5), (16.5, 3.5), (16.5, 4.5), (17.5, 4.5), (17.5, 4.5), (18.5, 4.5),
                   (18.5, 4.5), (19.5, 4.5), (19.5, 2.5), (20.5, 2.5), (20.5, 2.5), (21.5, 2.5), (21.5, 3.5),
                   (22.5, 3.5), (22.5, 3.5), (23.5, 3.5), (23.5, 2.5), (24.5, 2.5), (24.5, 2.5), (25.5, 2.5),
                   (25.5, 1.5), (26.5, 1.5), (26.5, 0.5), (27.5, 0.5), (27.5, 3.5), (28.5, 3.5), (28.5, 3.5),
                   (29.5, 3.5), (29.5, 3.5), (30.5, 3.5), (30.5, 2.5), (31.5, 2.5), (31.5, 2.5), (32.5, 2.5),
                   (32.5, 2.5), (33.5, 2.5), (33.5, 2.5), (34.5, 2.5), (34.5, 2.5), (35.5, 2.5), (35.5, 1.5),
                   (36.5, 1.5), (36.5, -0.5), (37.5, -0.5), (37.5, 2.5), (38.5, 2.5), (38.5, 2.5), (39.5, 2.5),
                   (39.5, 2.5), (40.5, 2.5), (40.5, 2.5), (41.5, 2.5), (41.5, 3.5), (42.5, 3.5), (42.5, 4.5),
                   (43.5, 4.5), (43.5, 4.5), (44.5, 4.5), (44.5, 4.5), (45.5, 4.5), (45.5, 4.5), (46.5, 4.5),
                   (46.5, 4.5), (47.5, 4.5), (47.5, 4.5), (48.5, 4.5), (48.5, 3.5), (49.5, 3.5), (49.5, 3.5),
                   (50.5, 3.5), (50.5, 3.5), (51.5, 3.5), (51.5, 4.5), (52.5, 4.5), (52.5, 4.5), (53.5, 4.5),
                   (53.5, 3.5), (54.5, 3.5), (54.5, 3.5), (55.5, 3.5), (55.5, 3.5), (56.5, 3.5), (56.5, 2.5),
                   (57.5, 2.5), (57.5, 2.5), (58.5, 2.5), (58.5, 2.5), (59.5, 2.5), (59.5, 2.5), (60.5, 2.5),
                   (60.5, 4.5), (61.5, 4.5), (61.5, 3.5), (62.5, 3.5), (62.5, 3.5), (63.5, 3.5), (63.5, 3.5),
                   (64.5, 3.5), (64.5, 3.5), (65.5, 3.5), (65.5, 3.5), (66.5, 3.5), (66.5, 3.5), (67.5, 3.5),
                   (67.5, 3.5), (68.5, 3.5), (68.5, 4.5), (69.5, 4.5), (69.5, 4.5), (70.5, 4.5), (70.5, 4.5),
                   (71.5, 4.5), (71.5, 3.5), (72.5, 3.5), (72.5, 3.5), (73.5, 3.5), (73.5, 2.5), (74.5, 2.5),
                   (74.5, 2.5), (75.5, 2.5), (75.5, 2.5), (76.5, 2.5), (76.5, 2.5), (77.5, 2.5), (77.5, 1.5),
                   (78.5, 1.5), (78.5, 4.5), (79.5, 4.5), (79.5, 3.5), (80.5, 3.5), (80.5, 3.5), (81.5, 3.5),
                   (81.5, 2.5), (82.5, 2.5), (82.5, 2.5), (83.5, 2.5), (83.5, 2.5), (84.5, 2.5), (84.5, 2.5),
                   (85.5, 2.5), (85.5, 2.5), (86.5, 2.5), (86.5, 2.5), (87.5, 2.5), (87.5, 2.5), (88.5, 2.5),
                   (88.5, -0.5), (89.5, -0.5), (89.5, -0.5), (90.5, -0.5), (90.5, -0.5), (91.5, -0.5), (91.5, -0.5),
                   (92.5, -0.5), (92.5, 1.5), (91.5, 1.5), (91.5, 4.5), (92.5, 4.5), (93.5, 4.5), (93.5, 4.5),
                   (94.5, 4.5), (94.5, 3.5), (95.5, 3.5), (95.5, 3.5), (96.5, 3.5), (96.5, 4.5), (97.5, 4.5),
                   (97.5, 4.5), (98.5, 4.5), (98.5, 4.5), (99.5, 4.5), (99.5, 4.5), (100.5, 4.5), (100.5, 4.5),
                   (101.5, 4.5), (101.5, 4.5), (102.5, 4.5), (102.5, 4.5), (103.5, 4.5), (103.5, 3.5), (104.5, 3.5),
                   (104.5, 3.5), (105.5, 3.5), (105.5, 3.5), (106.5, 3.5), (106.5, 4.5), (107.5, 4.5), (107.5, 4.5),
                   (108.5, 4.5), (108.5, 4.5), (109.5, 4.5), (109.5, 4.5), (110.5, 4.5), (110.5, 4.5), (111.5, 4.5),
                   (111.5, 4.5), (112.5, 4.5), (112.5, 4.5), (113.5, 4.5), (113.5, 4.5), (114.5, 4.5), (114.5, 4.5),
                   (115.5, 4.5), (115.5, 4.5), (116.5, 4.5), (116.5, 4.5)]

        xs, ys = zip(*polygon)

        fig = plt.figure(figsize=(14, 6))
        ax = plt.subplot2grid((1, 3), (0, 0), colspan=3)

        activations = activations[:-5]  # The E... electrodes are noise electrodes and can therefore be discarded

        im = ax.imshow(activations.T, aspect='auto', origin='lower', cmap='Reds', interpolation='None', vmin=0,
                           vmax=vmax)

        ax.set_yticks(np.arange(0, 5))
        ax.grid(False)

        ttl = ax.set_title('Electrode Shaft', fontdict={'fontsize': 12, 'fontweight': 'bold'})
        ttl.set_position([.5, 1.06])
        ax.plot(xs, ys, color='black', linestyle=':', linewidth=1)  # Plot feature selection boundary
        ax.set_yticklabels(reversed(['-200', '-150', '-100', '-50', '0']))
        ax.set_ylabel('Temporal Context [in ms]')
        ax.set_xticks([])

        # Draw electrode color patches (manual)
        rect_x = [-0.5, 7.5, 19.5, 27.5, 37.5, 42.5, 60.5, 78.5, 88.5, 96.5, 105.5, 116.5]
        rect_width = [8, 12, 8, 10, 5, 18, 18, 10, 8, 9, 11, ]
        for (x, w, ci) in zip(rect_x, rect_width, np.arange(len(rect_x))):

            h = 4.51
            xy = np.array([[x, x + w, x + w], [h, h, h + 0.3]]).T
            rect = patches.Polygon(xy, linewidth=1, clip_on=False, fill=True,
                                     edgecolor=self.tab10_modified[ci], facecolor=self.tab10_modified[ci])
            ax.add_patch(rect)

        ax.set_xlim(-0.5, len(activations) - 0.5)
        ax.tick_params(axis='x', length=5)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Draw colorbar next to activation plot
        cbaxes = fig.add_axes([0.94, 0.03, 0.025, 0.85])
        cb = plt.colorbar(im, cax=cbaxes, ticks=[0, vmax])
        cbaxes.yaxis.set_ticks_position('right')
        cb.ax.set_yticklabels(['0', '1.75'])
        cb.set_label('Average Model Weights', rotation=270, labelpad=-5)
        plt.subplots_adjust(left=0.06, bottom=0.03, top=0.88, right=0.93)
        plt.savefig(filename, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Execute Experiment 1.')
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

    logging.getLogger('decoder.py').setLevel(logging.WARNING)
    logging.getLogger('ECoGFeatCalc.py').setLevel(logging.WARNING)

    session_dir = os.path.join(config['General']['storage_dir'], config['General']['session'])
    dest_dir = os.path.join(config['General']['temp_dir'], config['General']['session'])

    logger.info('Config file: {}'.format(args.config))
    logger.info('Session: {}'.format(config['General']['session']))
    logger.info('Session dir: {}'.format(session_dir))
    logger.info('Dest dir: {}'.format(dest_dir))

    # Run Experiment 4
    exp4 = Experiment4(session_dir=session_dir)
    activations = exp4.compute_activations()
    exp4.plot_results(activations=activations, filename=os.path.join(dest_dir, 'activation_map.png'))
    logger.info('Finished experiment 4.')
