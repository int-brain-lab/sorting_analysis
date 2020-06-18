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
    paths : dict
        Relevant paths to files containing data that is loaded into the object.
    ts : ndarray
        Timestamps for all spikes.
    cid : ndarray
        Cluster ids for all spikes.
    template_wfs : ndarray
        Waveforms for each unit's template.
    wfs : ndarray
        Spike Waveforms.
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
                'name' : str (optional)
                    A name used to identify the object.
                'path_to_phy_files' : str OR Path (optional)
                    Path to the sorter's output directory that contains phy-friendly filenames.
                'path_to_alf_files' : str OR Path (optional)
                    Path to alf output directory.
                'path_to_ts' : str OR Path (optional)
                    Path to spike timestamps (in s) array.
                'path_to_cid' : str OR Path (optional)
                    Path to spike cluster ids array.

        Examples
        --------
        Set-up:
            >>> from pathlib import Path
            >>> from sorting_analysis.compare.process import SorterOutput

        1) Create generic object from saved .npy arrays
            >>> params = {'name': 's', 'path_to_ts': Path('/path/to/ts.npy'), 'path_to_cid': Path('/path/to/cid.npy')}
            >>> s = SorterOutput(params)

        2) Create object from saved ks2 or yass output (in phy-friendly directory)
            >>> params_ks2 = {'name': 's_ks2', 'path_to_phy_files': Path('/path/to/ks2out/')}
            >>> s_ks2 = SorterOutput(params_ks2)
            >>> params_yass = {'name': 's_yass', 'path_to_phy_files': Path('/path/to/yassout/')}
            >>> s_yass = SorterOutput(params_yass)

        3) Create object from output saved in alf directory.
            >>> params_alf = {'name': 's_alf', 'path_to_alf_files': Path('/path/to/alf/')}
            >>> s_alf = SorterOutput(params_alf)
        """

        # Determine which files to load from based on sorter output.
        param_keys = list(params.keys())
        self.name = params['name'] if 'name' in param_keys else None
        # generic sorting output
        if not('path_to_phy_files' in param_keys) and not('path_to_alf_files' in param_keys):
            files = [Path(params['path_to_ts']), Path(params['path_to_cid'])]
        # sorting output saved for phy
        elif 'path_to_phy_files' in param_keys:
            d = Path(params['path_to_phy_files'])
            assert Path.exists(d), FileNotFoundError(f'Cannot find phy directory ({d})')
            files = [d.joinpath('spike_times.npy'), d.joinpath('spike_clusters.npy'), d.joinpath('templates.npy')]
        # sorting output saved in alf format
        elif 'path_to_alf_files' in param_keys:
            # check files can be found
            d = Path(params['path_to_alf_files'])
            assert Path.exists(d), FileNotFoundError(f'Cannot find alf directory ({d})')
            files = [d.joinpath('spikes.times.npy'), d.joinpath('spikes.clusters.npy'),
                     d.joinpath('clusters.waveforms.npy')]

        # Check that files can be found and load em.
        for f in files:
            assert f.exists(), FileNotFoundError(f'Cannot find {f}')
        self.ts = np.load(files[0])
        self.cid = np.load(files[1])
        self.template_wfs = np.load(files[2]) if len(files) > 2 else None


if __name__ == '__main__':
    params = {'name': 's_ks', 'path_to_phy_files': Path('/home/jai/yass_analysis/KS2_sorted_results')}
    s_ks = SorterOutput(params)
    params = {'name': 's_ks_gen', 'path_to_ts': Path('/home/jai/yass_analysis/KS2_sorted_results/spike_times.npy'),
              'path_to_cid': Path('/home/jai/yass_analysis/KS2_sorted_results/spike_times.npy')}
    s_ks_gen = SorterOutput(params)
    # s_yass =
    # s_yass_gen =
