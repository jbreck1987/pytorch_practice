"""
This module contains functions/classes that can be used to manipulate the training data
for different model architectures and experiments.
"""
import numpy as np
import numpy
import math

def create_windows(i: numpy.array,
                   q: numpy.array,
                   photon_arrivals: numpy.array,
                   with_pulses: list,
                   no_pulses: list,
                   num_samples: int,
                   no_pulse_fraction: float,
                   window_size=150):
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
        # Check to see if any of the pulse indices are in this window
        if np.sum((pulse_indices >= window) & (pulse_indices < window + 150)) > 0:
            # If so add the window to the with_pulses container
            with_pulses.append(np.vstack((i[window : window + 150],
                                          q[window : window + 150],
                                          photon_arrivals[window : window + 150])).reshape(3, window_size)) # Reshaping to get in nice form for CNN
        # If no pulses are in the window and the no-pulse fraction hasn't been met,
        # add to the no_pulses container
        elif len(no_pulses) < num_samples * no_pulse_fraction:
            no_pulses.append(np.vstack((i[window : window + 150],
                                        q[window : window + 150],
                                        photon_arrivals[window : window + 150])).reshape(3, window_size)) # Reshaping to get in nice form for CNN
 
    
        # Continue to next window doing nothing since threshold for no_pulse samples has been met and the window lacks a pulse
        else:
            continue