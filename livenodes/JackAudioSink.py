#!/usr/bin/env python3
"""
Multithreaded jack audio sink

block_size as small as your computer will let you go (and smaller than blocks returned by
synthesis, post-resampling), max_pipe_blocks as high as you need for reasonably low xrun count.
"""

import jack
import numpy as np
import threading
import multiprocessing
import logging
import time
import samplerate

from . import Node

class JackAudioSink(Node.Node):
    def __init__(self, orig_sample_rate, block_size = 128, max_pipe_blocks = 8, wait_for_blocks = 3, allow_fractional_resample = False, name = "JackAudioSink"):
        super(JackAudioSink, self).__init__(has_outputs = False, name = name)
        
        # Store settings
        self.block_size = block_size   
        self.wait_for_blocks = wait_for_blocks
        self.orig_sample_rate = orig_sample_rate
        self.allow_fractional_resample = allow_fractional_resample
        
        # IPC setup
        self.samplePipeOut, self.samplePipeIn = multiprocessing.Pipe(False)
        self.pipe_fill = multiprocessing.Value('i', 0)
        self.max_pipe_blocks = max_pipe_blocks
        
        self.client_reset()
        
    def client_reset(self):
         # Jack setup
        self.client = jack.Client('JackAudioSink')
        self.client.blocksize = self.block_size
        self.tmp_buf = np.array([0.0] * self.block_size)
        self.tmp_buf_pos = 0
        self.is_active = False
        
        # Debug
        self.start_time = 0
        self.sample_count = 0
        self.xrun_count = 0
        
        self.port = self.client.outports.register('audio_out')
        
        # Make sure sample rate works and set multiplier
        if not self.allow_fractional_resample:
            if self.client.samplerate % self.orig_sample_rate != 0:
                raise(ValueError("OS sample rate " + str(self.client.samplerate) + " must be evenly divisible by given sample rate " + str(self.orig_sample_rate)))
            self.sample_multiplier = int(self.client.samplerate / self.orig_sample_rate)
        else:
            self.sample_multiplier = float(self.client.samplerate / self.orig_sample_rate)
        self.resampler = samplerate.Resampler('sinc_fastest', channels=1)
    
        # Callback setup
        self.client.set_process_callback(self.process)
        self.client.set_xrun_callback(self.xrun)
        
    def process(self, blocksize):
        if self.samplePipeOut.poll():
            self.sample_count += self.block_size
            self.port.get_array()[:] = self.samplePipeOut.recv()
        
        with self.pipe_fill.get_lock():
            self.pipe_fill.value -= 1
            
    def xrun(self, usecs):
        self.xrun_count += 1
        if self.xrun_count % 50 == 0:
            logging.info("xruns: " + str(self.xrun_count) + ", samples: " + str(self.sample_count) + ", time: " + str(time.time() - self.start_time))
    
    def get_stats(self):
        return([self.pipe_fill.value, self.xrun_count, self.sample_count, time.time() - self.start_time])
    
    def start_processing(self, recurse = True):
        """
        Wait for first sample, then start running
        """
        if self.is_active == False:
            """
            while True: # TODO turned this off because it seemed to work okay without
                print("Wait for first sample...")
                self.samplePipeOut.poll(None)
                with self.pipe_fill.get_lock():
                    if self.pipe_fill.value >= self.wait_for_blocks:
                        break
            """
            
            self.start_time = time.time()
            self.client.activate()
            
            # Connect mono -> stereo
            target_ports = self.client.get_ports(is_physical = True, is_input = True, is_audio = True)
            self.port.connect(target_ports[0])
            self.port.connect(target_ports[1])
            self.is_active = True
        super(JackAudioSink, self).start_processing(recurse)
        
    def stop_processing(self, recurse = True):
        super(JackAudioSink, self).stop_processing(recurse)
        self.client.deactivate()
        self.client.close()
        self.is_active = False
        self.client_reset()
        
    def play_or_drop_block(self, block):
        send_block = False
        with self.pipe_fill.get_lock():
            if self.pipe_fill.value < self.max_pipe_blocks:
                send_block = True
                self.pipe_fill.value += 1
        if send_block == True:
            self.samplePipeIn.send(block)
    

    def add_data(self, samples, data_id = None):
        if samples is None:
            return
        
        samples_res = self.resampler.process(samples / (2**15), self.sample_multiplier)
            
        if len(samples_res) == 0:
            return
        
        # First buffer
        nextPos = self.block_size - self.tmp_buf_pos
        self.tmp_buf[self.tmp_buf_pos:] = samples_res[:nextPos]
        self.play_or_drop_block(self.tmp_buf)
        
        # Middle buffers
        samples_len = len(samples_res)
        while nextPos + self.block_size <= samples_len:
            self.play_or_drop_block(samples_res[nextPos:nextPos + self.block_size])
            nextPos += self.block_size
        
        # Last buffer
        self.tmp_buf[:samples_len - nextPos] = samples_res[nextPos:]
        self.tmp_buf_pos = samples_len - nextPos
