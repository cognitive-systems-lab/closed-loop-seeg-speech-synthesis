# Closed-Loop Speech Synthesis

This repository contains the code from the paper:  
**Real-time Synthesis of Imagined Speech Processes from Minimally Invasive Recordings of Neural Activity** 
by Miguel Angrick, Maarten Ottenhoff, Lorenz Diener, Darius Ivucic, Gabriel Ivucic, Sofoklis Goulis, Jeremy Saal, Albert J. Colon, Louis Wagner, Dean J. Krusienski, Pieter L. Kubben, Tanja Schultz and Christian Herff.

The approach enables a closed-loop decoding of audible speech based on invasive neural signals. 
Incoming neural signals are converted into an acoustic speech signal in real-time, which gets played back to the patient 
as a continous neurofeedback. 

## Installation
We use Python3.6 as our programming language. All additional required packages are listed in the requirements.txt file with appropriate version numbers. 
In addition, it is neccessary to download the python library for importing Extendable Data Format (XDF), which can be found [here](https://github.com/xdf-modules/pyxdf). We have used the version 1.15.0. The important part is the pyxdf.py file, which needs to be stored in the local directory (which can be found in the root directory of the code) as xdf.py.

## Configuration files
Hyperparameter settings for the different scripts are organized in configuration files with the **.ini** extension.
Each configuration file has several sections regarding the different script files.
In addition, the **General** section contains two very important parameters, namely the storage directory and the session name, since they are commonly used across multiple script files.

Here is an example with all available parameters and its explanations:
```ini 
[General]
; Storage_dir contains the path where the experiment files are being stored (from training, decoding, ...).
storage_dir = ./subjects
; Name of the session corresponds to the folder name created in storage_dir.
session = kh13

[Training]
; Point to the XDF file containing the training data.
file = C:/Users/miguel/Datasets/speech_kh13.xdf
; Specify at which frequency is the power line noise.
power_line = 50
; Specify which channels should be used. Supports Regex expressions.
channels = [A-Za-z0-9]*$
; Specify if the interactive plot from MNE with the channels should be opened.
; This can be useful to check the selected channels and manually remove broken ones.
show_interactive_channel_view = False
; If the same experiment is being rerun, shall the files be overwritten?
; Otherwise the script throws an exception.
overwrite_on_rerun = True
; Draw plots from training
draw_plots = True

[Decoding]
; LSL stream name of the sEEG data
stream_name = dev_sEEG
; LSL stream name of the experiment markers
marker_stream_name = SingleWordsMarkerStream
; Norm factor for the griffin-lim reconstruction. This one might be manually be adjusted.
griffin_lim_norm = 10
; Name of the decoding run (for example to separate audible and silent).
; This will create a new folder in the session directory named <run> and
; store the appropriate files there.
run = whisper
; If the same decoding run is being repeated, shall the files be overwritten?
; Otherwise the script throws an exception.
overwrite_on_rerun = True

[Development]
; Point to the XDF file loaded by the dev_lsl_streamer for sending real sEEG data.
file = C:/Users/miguel/Datasets/speech_kh13.xdf
```
**It is recommended that all parameters are set accordingly for the specific script since they might be used**

Important notes on some crucial parameters: 
* storage_dir is the root path in which the directory **session** will be created. 
* A session corresponds to one training. So all the training files will be stored there. If you rerun the training, for example with a different set of electrodes, the files from the previous training might get **overwritten**.
* The decoding runs are stored in the session folder to which training they belong. Each decoding run creates a new directory based on the given name inside the session folder.
* Script files like **train.py**, **decode.py**, dev_lsl_streamer.py, ... accept the path to the configuration file **as a first parameter**.
* Several parameters from the configuration file can be overwritten in the command line options. For example: Assume the **run** parameter in the configuration file is set to **imagine**. Than, a new decoding can be introduced with: 
```bash 
python3 decode.py config.ini --run imagine 
```
* Following up on the last bullet: All configured parameters **stay the same**, except for the decoding run directoy name. 

## Training

The training script **train.py** is used to train the linear models (LDAs based on quantized logMel spectral coefficients).
An example call has the following form:
```bash 
>>> python3 train.py config.ini
```
The training script needs the parameters to be set from the **General** and the **Training** section from the configuration file. 
Important notes on some crucial parameters: 
* the **file** parameter accepts either XDF or HDF5 files
* In case of HDF5 file it needs the following datasets: ecog, audio, ecogSR, audioSR
* the **channels** parameter accepts a list of regular expression. Here are some Examples:
```ini 
--channels LA.*  ; Select all electrodes starting with LA
--channels LA.* RA.*  ; Select all electrodes starting either with LA or RA
--channels [A-Za-z0-9]*$  ; Select all electrodes, except for the ones like EKG+, PULSE+, ...
```
*  In case the **show_interactive_channel_view** is set to True, MNEs visualization tool will be opened showing all channels and highlighting in red which channels to exclude so far. By clicking **on** their channel names the selection can be manually modified. After window closing (with ESC or close button), the training process is started.
* The training script will store an configuration file in the session folder which is a copy of the one used for training and with updated parameters based on the ones overwritten on the command line. Therefore, the exact training can be repeated. 

The following files will be created by **train.py**:
* train.log &rarr; log file of the training
* LDAs.pkl &rarr; the list of trained LDA models
* params.h5 &rarr; required information for the decoding, like LDAs, bad channels, dequantization information.
* trainsset.png &rarr; Visualization of the training data for visual inspection
* training_features.npy &rarr; The training features for later use to compute activation maps.

## Decoding

The decoding script requires that a stream with sEEG data is running in LSL. The corresponding sample rate will extracted from the stream metadata.

An example on how to start the decoding:
```bash 
>>> python3 decode.py config.ini
```

The training script needs the parameters to be set from the General and the Decoding section from the configuration file. Important notes on some crucial parameters:

* The run folder will be placed inside the session folder, which contains the trained models.
* It might be necessary to manually adjust the gl_norm parameter (default is 10). 

In addition, the decoding script can connect to a marker stream. The stream name needs to be defined upfront. If there is no marker stream available, the decoding will still work on the sEEG data. As soon as the marker script is started, the decoding script will connect to it and start a listener to write the markers to a file.

The following files will be created by **decode.py***:
* audio.wav &rarr; the decoded audio output send to the loudspeaker
* decode.ini &rarr; configuration of the decoding run
* decode.log &rarr; log file from decoding
* decoding.png &rarr; time aligned decoded spectrogram and audio
* first_timestamp.npy &rarr; first timestamp received from the sEEG stream 
* markers.csv &rarr; **Optional** marker timestamps
* sEEG.hdf &rarr; contains the streamed sEEG data received from LSL
* spectrogram.npy &rarr; the reconstructed spectrogram from the linear models

## Dev LSL Streamer
The python script dev_lsl_streamer.py can be used to stream sEEG over LSL for debugging purposes. It can handle XDF and HDF files as input. The input file needs to be specified in an ini file under the section Development. The corresponding parameter is **file**.
Example:
```bash 
>>> python3 dev_lsl_streamer.py config.ini 
```

## Evaluation Steps
The evaluation steps are stored in the folder **eval_steps**. 
Many evaluation steps require **.ini** file which contains the necessary informations to run the analysis.
In general, a configuration file looks like this:
```ini
[General]
; Storage_dir contains the path where the experiment session is being stored.
storage_dir = /home/miguel/recordings/closed_loop_experiments/
; Session name
session = kh13
; destination root directory (will create a new folder with the session name, this is the place where the results will be stored).
temp_dir = /home/miguel/evaluation/closed_loop_analysis/

[Experiment1]
nb_randomization_runs = 100
griffin_lim_norm = 10

[Experiment2]
; Which evaluation should be computed. Options are "pm_only", "chance_only" and "both".
which = both
other_xdf = exec1.xdf,followthedot.xdf,imag1.xdf
nb_randomization_runs = 1000
decoding_runs = whisper,imagine
griffin_lim_norm = 10

[Experiment3]
decoding_runs = whisper,imagine
vad_energy_threshold = 0.5
vad_energy_mean_scale = 1
vad_frames_context = 5
vad_proportion_threshold = 0.6
```
The list below gives a short summary about each evaluation step:
1. **extract_trials**: A script for extracting the waveforms for each trial in a session and a decoding run. In addition, it outputs **.lab** files which contain the timings of trial boundaries in a common format for storing labels. 
2. **exp1**: Experiment 1 is used to quantify the performance of the proposed decoding approach. The script reconstructs the complete speech spectrogram in 5-fold cross validation and, in addition, estimates a chance level. The results can be visualized by subsequently running the **figure3.py** script, which plots the Figure 3 from the manuscript.
3. **exp2**: Experiment 2 is used to quantify the performance based on the whisperes and imagined communication modality. The script calculates the DTW correlations which are shown in Figure 4.C in the manuscript.
4. **exp3**: Experiment 3 is used to compute the proportion of decoded speech during trials and during resting phases. It uses an energy based voice activity detection.
5. **exp4**: Experiment 4 generates an average activation map with respect to the anatomical contributions.
6. **figure4**: figure4.py uses the intermediate results from exp2.py and exp3.py to plot Figure 4 from the manuscript. 

## Steps to reproduce results

The open-loop and closed-loop recordings of this study are publicly available. To reproduce the results from the paper, the evaluation steps have to be repeated. For this, some paths from the experiment.ini file have to be adapted:  
In the General-section, the combined path between *storage_dir* and *session* have to point to the data files. In addition, the *temp_dir* has to point to a new directory, in which the results are written to disc. 
Following on that, the evaluation steps can be executed by:
```python
python3 exp1.py ../config/evaluation.ini
```
It is recommended to execute the steps in the following order: *extract_trials.py*, *exp1.py*, *exp2.py*, *exp3.py*, *exp4.py*, *figure_3.py* and *figure_4.py*.
