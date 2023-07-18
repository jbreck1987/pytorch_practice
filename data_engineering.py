"""
This module contains functions/classes that can be used to manipulate the training data
for different model architectures and experiments.
"""
import numpy as np
import numpy
import math

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
                   num_samples: int,
                   no_pulse_fraction: float,
                   edge_padding: int,
                   window_size: int) -> None:
    """
    This function takes the output of the mkidreadoutanalysis objects (I, Q, and Photon Arrival vectors) and chunks it into smaller arrays. The code
    also separates chunks with photons and those without photons with the goal of limiting the number of samples
    without photon pulses since there are vastly more windows in the synthetic data in this category. It uses "scanning" logic to scan over the full
    arrays with a given window size and inspects that window for a photon event. The window is then added to the appropriate container (photon/no photon).
    """

    # First determine the last index in the scanning range (need to have length of photon arrivals array be multiple of window_size)
    end_index = math.floor(len(photon_arrivals) / window_size) * window_size

    # Determine all the indicies in the i,q, photon_arrvial vecs denoting a pulse
    pulse_indices = np.argwhere(photon_arrivals == 1).squeeze()

    # Now scan across the photon arrival vector and look at windows of length window_size with and without photon events
    for window in range(0, end_index - window_size + 1, window_size):
        # Check to see if any of the pulse indices are in this window and
        if np.sum((pulse_indices >= window + edge_padding) & (pulse_indices < window + window_size - edge_padding)) > 0:
            # If so add the window to the with_pulses container
            with_pulses.append(np.vstack((i[window : window + window_size],
                                          q[window : window + window_size],
                                          photon_arrivals[window : window + window_size])).reshape(3, window_size)) # Reshaping to get in nice form for CNN
        # If no pulses are in the window and the no-pulse fraction hasn't been met,
        # add to the no_pulses container
        elif len(no_pulses) < num_samples * no_pulse_fraction:
            no_pulses.append(np.vstack((i[window : window + window_size],
                                        q[window : window + window_size],
                                        photon_arrivals[window : window + window_size])).reshape(3, window_size)) # Reshaping to get in nice form for CNN
 
    
        # Continue to next window doing nothing since threshold for no_pulse samples has been met and the window lacks a pulse
        else:
            continue

def make_dataset(qp_timestream: QuasiparticleTimeStream,
                 num_samples: int,
                 no_pulse_fraction: float,
                 with_pulses: list,
                 no_pulses: list,
                 cps=500,
                 edge_padding=0,
                 window_size=150,
            ) -> None:
    # Generate the training set in the following format: [np.array([i,q,label]), ...] where i,q,label are all WINDOW_SIZE length arrays.
    # Each list element is a 3 x 1 x WINDOW_SIZE numpy array.
    while len(with_pulses) < num_samples - (num_samples * no_pulse_fraction):
        # We want the training data to be varied, so lets use the Poisson sampled
        # gen_photon_arrivals method to change the photon flux per iteration
        photon_arrivals = qp_timestream.gen_photon_arrivals(cps=cps)
        qp_timestream.populate_photons() # this is necessary for the resonator object in the gen_iq function
        I, Q = gen_iq(qp_timestream)
        create_windows(I,
                       Q,
                       photon_arrivals,
                       with_pulses,
                       no_pulses,
                       num_samples,
                       no_pulse_fraction,
                       window_size=window_size,
                       edge_padding=edge_padding)
    print(f'Number of samples with pulses: {len(with_pulses)}')
    print(f'Number of samples without pulses: {len(no_pulses)}')