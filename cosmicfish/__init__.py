#from .forecast import analysis, v_effective_table_generator, k_table_generator, primordial_table_generator
#from .data import spectrum
#from .io import is_data, is_data2, correct_path, generate_data, check_data, install_class

from .forecast import relic_convergence_analysis
from .data import spectrum, gen_v_eff, gen_k_table
from .io import install_class, generate_data, check_data, is_data, correct_path, config_directory, replace_text

