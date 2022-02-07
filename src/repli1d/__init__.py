# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound
__all__ = ['explore_param_yeast', 'detect_and_optimise_lgbis',
           'convert_Bw_to_csv', 'concat_and_rename_road',
           'whole_pipeline_extra', 'grid_search_simulate',
           'convert_Bw', 'whole_pipeline_from_data_file',
           'convert_5_to_1', 'hopt', 'detect', 'on_whole',
           'training', 'average_expe', 'extract_AT_hook',
           'visu_signals', 'prune_marks', 'detect_and_optimise',
           'nn_hopt', 'find_best', 'optimise', 'small_opty',
           'whole_pipeline', '__pycache__', 'models', 'fast_sim',
           'profile_generation', 'skeleton', 'retrieve_marks',
           'nn_exp_auto', 'convert_Bw_fromSR', 'nn_create_input',
           'detect_and_simulate', 'analyse_RFD', 'explode_benji',
           'expeData', 'pso', 'nn_create_input_roadmap',
           'grid_search_signal_opti', 'nn_exp', 'fast_sim_break',
           'tools', 'visu_browser', 'average', 'get_from_road',
           'nn', '__init__', 'single_mol_analysis',
           'get_highest_correlation', 'detect_and_optimise_lg',
           'convert_profile_to_bed', 'scipy_min2', 'scipy_min',
           'convert_Hadi_to_csv', 'development']
try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'repli1D'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound
