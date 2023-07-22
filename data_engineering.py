"""
This module contains functions/classes that can be used to manipulate the training data
for different model architectures and experiments.
"""
import numpy as np
import numpy
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import random
from typing import List

from mkidreadoutanalysis.quasiparticletimestream import QuasiparticleTimeStream
from mkidreadoutanalysis.resonator import Resonator, ResonatorSweep, FrequencyGrid, RFElectronics, ReadoutPhotonResonator

def gen_iq(qp_timestream: QuasiparticleTimeStream):
    '''
    Generate I and Q time streams using the mkidreadoutanalysis library

    Inputs: 
        QuasiPartcileTimeStream object with populated photons
    
    Returns: tuple of two numpy arrays containing the I and Q timestreams respectively.
    '''

    #Creating a resonator object
    resonator = Resonator(f0=4.0012e9, qi=200000, qc=15000, xa=1e-9, a=0, tls_scale=1e2)
    rf = RFElectronics(gain=(3.0, 0, 0), phase_delay=0, cable_delay=50e-9)
    freq = FrequencyGrid( fc=4.0012e9, points=1000, span=500e6)

    #Creating Photon Resonator Readout
    lit_res_measurment = ReadoutPhotonResonator(resonator, qp_timestream, freq, rf)
    lit_res_measurment.noise_on = True #toggle white noise and line noise
    lit_res_measurment.rf.noise_scale = 10 #adjust white noise scale

    #configure line noise for Resonator
    lit_res_measurment.rf.line_noise.freqs = ([60, 50e3, 100e3, 250e3, -300e3, 300e3, 500e3]) # Hz and relative to center of bin (MKID we are reading out)
    lit_res_measurment.rf.line_noise.amplitudes = ([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01])
    lit_res_measurment.rf.line_noise.phases = ([0, 0.5, 0,1.3,0.5, 0.2, 2.4])

    #Generating Synthetic Data for Output
    I = lit_res_measurment.normalized_iq.real
    Q = lit_res_measurment.normalized_iq.imag

    return I, Q 

def create_windows(i: numpy.array,
                   q: numpy.array,
                   photon_arrivals: numpy.array,
                   with_pulses: list,
                   no_pulses: list,
                   single_pulse: bool,
                   num_samples: int,
                   no_pulse_fraction: float,
                   edge_padding: int,
                   window_size: int,
                   ) -> None:
    """
    This function takes the output of the mkidreadoutanalysis objects (I, Q, and Photon Arrival vectors) and chunks it into smaller arrays. The code
    also separates chunks with photons and those without photons with the goal of limiting the number of samples
    without photon pulses since there are vastly more windows in the synthetic data in this category. It uses "scanning" logic to scan over the full
    arrays with a given window size and inspects that window for a photon event. The window is then added to the appropriate container (photon/no photon).
    """

    # First determine the last index in the scanning range (need to have length of photon arrivals array be multiple of window_size)
    end_idx = math.floor(len(photon_arrivals) / window_size) * window_size

    # Now scan across the photon arrival vector and look at windows of length window_size with and without photon events
    for window in range(0, end_idx - window_size + 1, window_size):
        window_pulses = np.sum(photon_arrivals[window : window + window_size] == 1)
        window_pulse_idxs = np.argwhere(photon_arrivals[window : window + window_size])
        valid_window_start = window + edge_padding
        valid_window_end = window + window_size - edge_padding
        valid_pulse = ((window_pulse_idxs > valid_window_start) & (window_pulse_idxs < valid_window_end)).all()

        # If there are more than one pulses in the entire window and we only want single pulse data, skip this window
        if window_pulses > 1 and single_pulse:
            continue

        # Now any window with a pulse is valid as long as the pulse(s) dont live in the edge padding area
        elif window_pulses > 0 and valid_pulse:
            # If so add the window to the with_pulses container
            with_pulses.append(np.vstack((i[window : window + window_size],
                                          q[window : window + window_size],
                                          photon_arrivals[window : window + window_size])).reshape(3, window_size)) # Reshaping to get in nice form for CNN

        # If no pulses are in the window and the no-pulse fraction hasn't been met,
        # add to the no_pulses container
        elif len(no_pulses) < num_samples * no_pulse_fraction and window_pulses == 0:
            no_pulses.append(np.vstack((i[window : window + window_size],
                                        q[window : window + window_size],
                                        photon_arrivals[window : window + window_size])).reshape(3, window_size)) # Reshaping to get in nice form for CNN

def make_dataset(qp_timestream: QuasiparticleTimeStream,
                 num_samples: int,
                 no_pulse_fraction: float,
                 with_pulses: list,
                 no_pulses: list,
                 single_pulse: bool,
                 cps=500,
                 edge_padding=0,
                 window_size=150
            ) -> None:
    # Generate the training set in the following format: [np.array([i,q,label]), ...] where i,q,label are all WINDOW_SIZE length arrays.
    # Each list element is a 3 x 1 x WINDOW_SIZE numpy array.
    count = len(with_pulses)
    while len(with_pulses) < num_samples - (num_samples * no_pulse_fraction):
        # We want the training data to be varied, so lets use the Poisson sampled
        # gen_photon_arrivals method to change the photon flux per iteration
        photon_arrivals = qp_timestream.gen_photon_arrivals(cps=cps, seed=None)
        qp_timestream.populate_photons() # this is necessary for the resonator object in the gen_iq function
        I, Q = gen_iq(qp_timestream)
        create_windows(I,
                       Q,
                       photon_arrivals,
                       with_pulses,
                       no_pulses,
                       single_pulse,
                       num_samples,
                       no_pulse_fraction,
                       window_size=window_size,
                       edge_padding=edge_padding)
        # Give status update on number of samples with photons
        if len(with_pulses) > count:
            print(f'Num samples with photons: {len(with_pulses)}/{num_samples - (num_samples * no_pulse_fraction)}', end='\r')
            count = len(with_pulses)
    print(f'\nNumber of samples with pulses: {len(with_pulses)}')
    print(f'Number of samples without pulses: {len(no_pulses)}')


def plot_stream_data(i_stream: numpy.array,
                     q_stream: numpy.array,
                     label_stream: numpy.array,
                     units: str
                     ) -> None:
    """
    Plots the training data. Assumes training samples are I and Q timestreams
    and the labels are timestreams of photon events (based on the I/Q values).
    """

    _, ax = plt.subplots(3,1,figsize = (10,10))
    ax[0].plot(np.arange(0, i_stream.size), i_stream)
    ax[0].set_xlabel(f'Time ({units})')
    ax[0].set_ylabel('I Timestream', fontweight = 'bold', size = 'large')

    ax[1].plot(np.arange(0, q_stream.size), q_stream)
    ax[1].set_xlabel(f'Time ({units})')
    ax[1].set_ylabel('Q Timestream', fontweight = 'bold', size = 'large')

    ax[2].plot(np.arange(0, label_stream.size), label_stream)
    ax[2].set_xlabel(f'Time ({units})')
    ax[2].set_ylabel('Photon Timestream', fontweight = 'bold', size = 'large')
    plt.show()


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer) -> None:

    # Training Steps
    total_loss = 0 # need to reset loss every epoch
    for batch, (X, y) in enumerate(data_loader): # each batch has 32 data/labels, create object -> (batch, (X, y))
        model.train()
        y_pred = model(X) # Like before, need to get model's predictions
        loss = loss_fn(y_pred, y) # calculate loss for this batch
        total_loss += loss # add loss from this batch (mean loss of 32 samples) to total loss for the epoch (sum of all batch loss)

        # backprop step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # We want to see some updates within an epoch
        print(f'Batches processed: {batch + 1}/{len(data_loader)}, Samples processed: {(batch + 1) * data_loader.batch_size}/{len(data_loader.dataset)}', end='\r')
    
    # Now we want to find the AVERAGE loss and accuracy of all the batches
    total_loss /= len(data_loader)
    print('\n-----------')
    print(f'Mean Train Loss: {total_loss:.4f}')


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module) -> None:

    # Test Steps
    model.eval()
    total_loss = 0 # need to reset loss every epoch
    with torch.inference_mode():
        for X, y in data_loader: # each batch has 32 data/labels, create object -> (batch, (X_train, y_train))
            y_pred = model(X) # Like before, need to get model's predictions
            loss = loss_fn(y_pred, y) # calculate loss for this batch
            total_loss += loss # add loss from this batch (mean loss of 32 samples) to total loss for the epoch (sum of all batch loss)

        # Now we want to find the AVERAGE loss and accuracy of all the batches
        total_loss /= len(data_loader)
    print(f'Mean Test Loss: {total_loss:.4f}')
    print('-----------\n')

def make_predictions(model: torch.nn.Module, samples: list) -> list:
    """
    Given a list of samples, returns a tensor with the prediction for each sample
    """
    model.eval()
    with torch.inference_mode():
        return [model(x) for x in samples]

def add_noise(pulse_list: List[numpy.array], range: float) -> None:
    """
    Adds uniform noise to photon arrival data. The pulse_list is expected to have the shape returned by
    the make_dataset function (with photon arrival data in dimension 2).

    Inputs:
        pulse_list: list of numpy arrays, each containing I/Q/photon timestream data
        range: max value of sampled values to add as noise
    """
    rng = np.random.default_rng() 
    for sample in pulse_list:
        # Save the index with the pulse
        pulse_idx = sample[2] == 1 
        # Add the noise 
        sample[2] = rng.random(sample[2].shape) * (range - 0)
        # Make sure the pulse is still 1 
        sample[2][pulse_idx] = 1

