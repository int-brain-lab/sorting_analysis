"""
Loads spike data necessary for creating spike sorter comparative visualizations via the code in `visualize.py`
"""

import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from numba import jit


class SorterOutput:
    """
    A notebook containing information on spike sorter outputs.

    Attributes
    ----------
    name : str
        A name used to identify the object.
    ts : ndarray
        Timestamps for all spikes.
    cid : ndarray
        Cluster ids for all spikes.
    template_wfs : ndarray
        Waveforms for each unit's template.
    wfs : ndarray
        Waveforms for each spike.
    phy_model : TemplateModel
        A phy TemplateModel object created from the sorter output.
    features : dict
        Arrays of spike features.
    """

    def __init__(self, params):
        """
        Loads all the available, relevant sorter output.

        Parameters
        ----------
        params : dict
            Parameters used for loading in the sorter output.
                'name' : str
                    A name used to identify the object.
                'path_to_ts' : str OR Path (optional)
                    Path to spike timestamps array.
                'path_to_cid' : str OR Path (optional)
                    Path to spike cluster ids array.
                'path_to_ks2out' : str OR Path (optional)
                    Path to kilosort2 output directory.
                'path_to_yassout' : str OR Path (optional)
                    Path to yass output directory.

        Examples
        --------
        Set-up:
            >>> from pathlib import Path
            >>> from sorting_analysis.compare.load import Sorter_Output

        1) Create generic object from saved .npy arrays
            >>> params = {'name': 's', 'path_to_ts': Path('/path/to/ts.npy'), 'path_to_cid': Path('/path/to/cid.npy')}
            >>> s = Sorter_Output(params)

        2) Create object from saved ks2 output
            >>> params = {'name': 's', 'path_to_ks2out': Path('/path/to/ks2out/')}
            >>> s_ks2 = Sorter_Output(params)

        3) Create object from saved yass output
            >>> params = {'name': 's', 'path_to_yassout': Path('/path/to/yassout/')}
            >>> s_yass = Sorter_Output(params)
        """

        # Determine which files to load from based on sorter.
        param_keys = list(params.keys())
        self.name = params['name'] if 'name' in param_keys else None
        if 'path_to_ks2out' in param_keys:     # ks2 sorter
            path_to_ks2out = Path(params['path_to_ks2out'])
            assert Path.exists(path_to_ks2out),FileNotFoundError(f'Path to ks2 out ({path_to_ks2out}) does not exist!')
            path_to_ts = path_to_ks2out.joinpath('spike_times.npy')
            assert Path.exists(path_to_ts), FileNotFoundError(f'Could not find spike timestamps file ({path_to_ts}) !')
            self.ts = np.load(path_to_ks2out.joinpath('spike_times.npy'))
            self.cid = np.load(path_to_ks2out.joinpath('spike_clusters.npy'))
            self.template_wfs = np.load(path_to_ks2out.joinpath('spike_templates.npy'))
        elif 'path_to_yassout' in param_keys:  # yass sorter
            ...
        else                                   # generic sorter


if __name__ == '__main__':
    params = {'name': 's_ks', 'path_to_ks2out': Path('/home/jai/yass_analysis/KS2_sorted_results')}
    s_ks = Sorter_Output(params)
    params = {'name': 's_ks_gen', 'path_to_ts': Path('/home/jai/yass_analysis/KS2_sorted_results/spike_times.npy'),
              'path_to_cid': Path('/home/jai/yass_analysis/KS2_sorted_results/spike_times.npy')}
    s_ks_gen = Sorter_Output(params)
    # s_yass =
    # s_yass_gen =
