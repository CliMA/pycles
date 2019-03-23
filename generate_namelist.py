import argparse
import json
import pprint
from sys import exit
import uuid
import ast


def main():
    parser = argparse.ArgumentParser(prog='Namelist Generator')
    parser.add_argument('case_name')
    # Optional Arguments for CGILS
    parser.add_argument('--perturbed_temperature', default='False',
                        help='Specify if perturbed temperature case is to be run (CGILS) as True/False')
    parser.add_argument('--control_subsidence', default='False',
                        help='Specify if control subsidence is to be used in perturbed runs (CGILS) as True/False')
    parser.add_argument('--zgils_location', default='False',
                        help='specify location (6/11/12)')

    args = parser.parse_args()

    case_name = args.case_name

    #Optional Arguments for CGILS
    is_p2 = ast.literal_eval(args.perturbed_temperature)
    is_ctl_omega = ast.literal_eval(args.control_subsidence)
    zgils_loc = ast.literal_eval(args.zgils_location)
    print(zgils_loc)

    if case_name == 'StableBubble':
        namelist = StableBubble()
    elif case_name == 'SaturatedBubble':
        namelist = SaturatedBubble()
    elif case_name == 'SullivanPatton':
        namelist = SullivanPatton()
    elif case_name == 'Bomex':
        namelist = Bomex()
    elif case_name == 'Soares':
        namelist = Soares()
    elif case_name == 'Soares_moist':
        namelist = Soares_moist()
    elif case_name == 'Gabls':
        namelist = Gabls()
    elif case_name == 'DYCOMS_RF01':
        namelist = DYCOMS_RF01()
    elif case_name == 'DYCOMS_RF02':
        namelist = DYCOMS_RF02()
    elif case_name == 'SMOKE':
        namelist = SMOKE()
    elif case_name == 'Rico':
        namelist = Rico()
    elif case_name == 'CGILS_S6':
        namelist = CGILS_S6(is_p2, is_ctl_omega)
    elif case_name == 'CGILS_S11':
        namelist = CGILS_S11(is_p2, is_ctl_omega)
    elif case_name == 'CGILS_S12':
        namelist = CGILS_S12(is_p2, is_ctl_omega)
    elif case_name == 'ZGILS':
        namelist = ZGILS(zgils_loc)
    elif case_name == 'TRMM_LBA':
        namelist = TRMM_LBA()
    elif case_name == 'ARM_SGP':
        namelist = ARM_SGP()
    elif case_name == 'GATE_III':
        namelist = GATE_III()
    else:
        print('Not a vaild case name')
        exit()

    write_file(namelist)


def SullivanPatton():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 32
    namelist['grid']['ny'] = 32
    namelist['grid']['nz'] = 32
    namelist['grid']['gw'] = 7
    namelist['grid']['dx'] = 160.0
    namelist['grid']['dy'] = 160.0
    namelist['grid']['dz'] = 64.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 7200.0

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = True
    namelist['tracers']['scheme'] = 'UpdraftTracers'
    namelist['tracers']['use_lcl_tracers'] = False
    namelist['tracers']['timescale'] = 15.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_Dry'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.02
    namelist['damping']['Rayleigh']['z_d'] = 500.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['TKE']
    namelist['stats_io']['frequency'] = 100.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['temperature','buoyancy_frequency','viscosity']

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'SullivanPatton'
    namelist['meta']['casename'] = 'SullivanPatton'

    return namelist


def SaturatedBubble():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 100
    namelist['grid']['ny'] = 5
    namelist['grid']['nz'] = 50
    namelist['grid']['gw'] = 5
    namelist['grid']['dx'] = 200.0
    namelist['grid']['dy'] = 200.0
    namelist['grid']['dz'] = 200.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.3
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 1000.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_SA'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'UniformViscosity'
    namelist['sgs']['UniformViscosity'] = {}
    namelist['sgs']['UniformViscosity']['viscosity'] = 0.0
    namelist['sgs']['UniformViscosity']['diffusivity'] = 0.0

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'None'

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['conditional_stats'] = {}

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['None']
    namelist['stats_io']['frequency'] = 100.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 100.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['meta'] = {}
    namelist['meta']['casename'] = 'SaturatedBubble'
    namelist['meta']['simname'] = 'SaturatedBubble'

    return namelist


def StableBubble():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 512
    namelist['grid']['ny'] = 7
    namelist['grid']['nz'] = 64
    namelist['grid']['gw'] = 7
    namelist['grid']['dx'] = 100.0
    namelist['grid']['dy'] = 100.0
    namelist['grid']['dz'] = 100.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 1000.0

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = True
    namelist['tracers']['scheme'] = 'UpdraftTracers'
    namelist['tracers']['use_lcl_tracers'] = False
    namelist['tracers']['timescale'] = 15.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_Dry'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'UniformViscosity'
    namelist['sgs']['UniformViscosity'] = {}
    namelist['sgs']['UniformViscosity']['viscosity'] = 75.0
    namelist['sgs']['UniformViscosity']['diffusivity'] = 75.0

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'None'

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['conditional_stats'] = {} 

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['None']
    namelist['stats_io']['frequency'] = 100.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 100.0
    namelist['fields_io']['diagnostic_fields'] = ['temperature','buoyancy_frequency']

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'StableBubble'
    namelist['meta']['casename'] = 'StableBubble'

    return namelist


def Bomex():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 64
    namelist['grid']['ny'] = 64
    namelist['grid']['nz'] = 75
    namelist['grid']['gw'] = 7
    namelist['grid']['dx'] = 100.0
    namelist['grid']['dy'] = 100.0
    namelist['grid']['dz'] = 100 / 2.5

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 21600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = True
    namelist['tracers']['scheme'] = 'UpdraftTracers'
    namelist['tracers']['use_lcl_tracers'] = False
    namelist['tracers']['timescale'] = 15.0

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_SA'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 600

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Cumulus','TKE']
    namelist['stats_io']['frequency'] = 100.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Bomex'
    namelist['meta']['casename'] = 'Bomex'

    namelist['initialization'] = {}
    namelist['initialization']['random_seed_factor'] = 1

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = True
    namelist['tracers']['scheme'] = 'UpdraftTracers'
    namelist['tracers']['use_lcl_tracers'] = False
    namelist['tracers']['timescale'] = 15.0

    return namelist

def Soares():
    # adopted from: "An eddy-diffusivity/mass-flux parametrization for dry and shallow cumulus convection",
    # By P. M. M. SOARES, P. M. A. MIRANDA, A. P. SIEBESMA and J. TEIXEIRA, Q. J. R. Meteorol. Soc. (2004)
    # modifications: qt initial profile and flux set to zero, since no dry thermodynamics without condensation given

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    # Soares (2004): domain size = 6400 x 6400 m, domain height = 3000 (?) m; dx = ?, dy = ?, dz = 20 m
    # Nieuwstadt: domain size = ?, domain height = 2400m; dx = dy = 60 m, dz = 50-60 m
    # IOP Paper, old code: domain size = 6400 x 6400 m, domain height = 3750 m
    namelist['grid']['nx'] = 8#256    # IOP
    namelist['grid']['ny'] = 8#256    # IOP
    namelist['grid']['nz'] = 150    # IOP
    namelist['grid']['gw'] = 3      # for 2nd order
    namelist['grid']['dx'] = 25.0   # IOP
    namelist['grid']['dy'] = 25.0   # IOP
    namelist['grid']['dz'] = 25.0   # IOP

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3    # seems to be 3 in all cases???
    namelist['time_stepping']['cfl_limit'] = 0.3    # default: 0.7; IOP: 0.3
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 8*3600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'       # 'constant' or 'variable', for Clausius Clapeyron calculation

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = True
    namelist['tracers']['scheme'] = 'UpdraftTracers'
    namelist['tracers']['use_lcl_tracers'] = False
    namelist['tracers']['timescale'] = 15.0

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_Dry'     # Bomex: 'None_SA'; options: 'None_Dry' (no qt as Progn. Var.), 'None_SA', 'SB_Liquid'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'  # seems to be this in all cases???

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] = {}
    namelist['sgs']['Smagorinsky']['cs'] = 0.17
    namelist['sgs']['UniformViscosity'] = {}
    namelist['sgs']['UniformViscosity']['viscosity'] = 1.2
    namelist['sgs']['UniformViscosity']['diffusivity'] = 3.6

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False      # seems to be set to False for all cases???

    # 2 = second_order_m
    # 32 = second_order_ml_m
    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 2
    # 2 = second_order_a
    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 2

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'  # no more 'DampingToDomainMean' ???
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.02
    namelist['damping']['Rayleigh']['z_d'] = 800.0  # ??? depth of damping layer?

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    # profile outputs
    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Flux']    # AuxiliaryStatistics
    namelist['stats_io']['frequency'] = 900.0

    # field outputs
    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['temperature','viscosity']   # defines diagnostic variable output fields (progn. variables output in restart files?!)

    # Conditional Statistics
    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Soares'
    namelist['meta']['casename'] = 'Soares'

    namelist['restart'] = {}
    namelist['restart']['output'] = False
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['visualization'] = {}
    namelist['visualization']['frequency'] = 1800.0

    namelist['stochastic_noise'] = {}
    namelist['stochastic_noise']['flag'] = True
    namelist['stochastic_noise']['amplitude'] = 0.05

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = True
    namelist['tracers']['scheme'] = 'UpdraftTracers'
    namelist['tracers']['use_lcl_tracers'] = False
    namelist['tracers']['timescale'] = 15.0

    return namelist



def Soares_moist():
    # adopted from: "An eddy-diffusivity/mass-flux parametrization for dry and shallow cumulus convection",
    # By P. M. M. SOARES, P. M. A. MIRANDA, A. P. SIEBESMA and J. TEIXEIRA, Q. J. R. Meteorol. Soc. (2004)
    # modifications: qt initial profile and flux set to zero, since no dry thermodynamics without condensation given
    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    # Soares (2004): domain size = 6400 x 6400 m, domain height = 3000 (?) m; dx = ?, dy = ?, dz = 20 m
    # Nieuwstadt: domain size = ?, domain height = 2400m; dx = dy = 60 m, dz = 50-60 m
    # IOP Paper, old code: domain size = 6400 x 6400 m, domain height = 3750 m
    namelist['grid']['nx'] = 256    # IOP
    namelist['grid']['ny'] = 256    # IOP
    namelist['grid']['nz'] = 150    # IOP
    namelist['grid']['gw'] = 3      # for 2nd order
    namelist['grid']['dx'] = 25.0   # IOP
    namelist['grid']['dy'] = 25.0   # IOP
    namelist['grid']['dz'] = 25.0   # IOP

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3    # seems to be 3 in all cases???
    namelist['time_stepping']['cfl_limit'] = 0.3    # default: 0.7; IOP: 0.3
    namelist['time_stepping']['dt_initial'] = 10.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 8*3600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'       # 'constant' or 'variable', for Clausius Clapeyron calculation

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = True
    namelist['tracers']['scheme'] = 'UpdraftTracers'
    namelist['tracers']['use_lcl_tracers'] = False
    namelist['tracers']['timescale'] = 15.0

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_SA'     # DCBL: 'None_Dry', Bomex: 'None_SA'; options: 'None_Dry' (no qt as Progn. Var.), 'None_SA', 'SB_Liquid'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'  # seems to be this in all cases???

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] = {}
    namelist['sgs']['Smagorinsky']['cs'] = 0.17
    namelist['sgs']['UniformViscosity'] = {}
    namelist['sgs']['UniformViscosity']['viscosity'] = 1.2
    namelist['sgs']['UniformViscosity']['diffusivity'] = 3.6
    namelist['sgs']['TKE'] = {}
    namelist['sgs']['TKE']['ck'] = 0.1
    namelist['sgs']['TKE']['cn'] = 0.76

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False      # seems to be set to False for all cases???

    # 2 = second_order_m
    # 32 = second_order_ml_m
    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 4
    # 2 = second_order_a
    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 4

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'  # no more 'DampingToDomainMean' ???
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.02
    namelist['damping']['Rayleigh']['z_d'] = 800.0  # ??? depth of damping layer?

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    # profile outputs
    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Fluxes']    # AuxiliaryStatistics
    namelist['stats_io']['frequency'] = 600.0

    # field outputs
    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['temperature','viscosity']   # defines diagnostic variable output fields (progn. variables output in restart files?!)

    # Conditional Statistics
    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Soares_moist'
    namelist['meta']['casename'] = 'Soares_moist'

    namelist['restart'] = {}
    namelist['restart']['output'] = False
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['visualization'] = {}
    namelist['visualization']['frequency'] = 1800.0

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = True
    namelist['tracers']['scheme'] = 'UpdraftTracers'
    namelist['tracers']['use_lcl_tracers'] = False
    namelist['tracers']['timescale'] = 15.0

    namelist['ClausiusClapeyron'] = {}
    namelist['ClausiusClapeyron']['temperature_min'] = 100.15
    namelist['ClausiusClapeyron']['temperature_max'] = 500.0

    return namelist


def Gabls():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 64
    namelist['grid']['ny'] = 64
    namelist['grid']['nz'] = 64
    namelist['grid']['gw'] = 7
    namelist['grid']['dx'] = 6.25
    namelist['grid']['dy'] = 6.25
    namelist['grid']['dz'] = 6.25

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] =1.0
    namelist['time_stepping']['dt_max'] = 2.0
    namelist['time_stepping']['t_max'] = 43200.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_Dry'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] ={}
    namelist['sgs']['Smagorinsky']['cs'] = 0.17
    namelist['sgs']['Smagorinsky']['prt'] = 1.0/3.0

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.02
    namelist['damping']['Rayleigh']['z_d'] = 100.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['StableBL']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['temperature','buoyancy_frequency','viscosity']

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Gabls'
    namelist['meta']['casename'] = 'Gabls'

    return namelist

def DYCOMS_RF01():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 96
    namelist['grid']['ny'] = 96
    namelist['grid']['nz'] = 300
    namelist['grid']['gw'] = 5
    namelist['grid']['dx'] = 35.0
    namelist['grid']['dy'] = 35.0
    namelist['grid']['dz'] = 5.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 4.0
    namelist['time_stepping']['t_max'] = 4.0 * 3600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_SA'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = False
    namelist['microphysics']['ccn'] = 100.0e6

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.002
    namelist['damping']['Rayleigh']['z_d'] = 500.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['DYCOMS', 'Flux','TKE']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 3600.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    namelist['visualization'] = {}
    namelist['visualization']['frequency'] = 1e6

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'DYCOMS_RF01'
    namelist['meta']['casename'] = 'DYCOMS_RF01'

    return namelist

def DYCOMS_RF02():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 128
    namelist['grid']['ny'] = 128
    namelist['grid']['nz'] = 300
    namelist['grid']['gw'] = 5
    namelist['grid']['dx'] = 50.0
    namelist['grid']['dy'] = 50.0
    namelist['grid']['dz'] = 5.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 6.0 * 3600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'SB_Liquid'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = True
    namelist['microphysics']['ccn'] = 55.0e6

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.002
    namelist['damping']['Rayleigh']['z_d'] = 500.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['DYCOMS', 'Flux']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 3600.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['visualization'] = {}
    namelist['visualization']['frequency'] = 1e6

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    namelist['meta'] = {}
    namelist['meta']['simname'] = 'DYCOMS_RF02'
    namelist['meta']['casename'] = 'DYCOMS_RF02'

    return namelist

def SMOKE():

    '''
    Namelist generator for the smoke cloud case:
    Bretherton, C. S., and coauthors, 1999:
    An intercomparison of radiatively- driven entrainment and turbulence in a smoke cloud,
    as simulated by different numerical models. Quart. J. Roy. Meteor. Soc., 125, 391-423. Full text copy.
    :return:
    '''


    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 64
    namelist['grid']['ny'] = 64
    namelist['grid']['nz'] = 50
    namelist['grid']['gw'] = 5
    namelist['grid']['dx'] = 50.0
    namelist['grid']['dy'] = 50.0
    namelist['grid']['dz'] = 25.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 4.0 * 3600.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_Dry'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7


    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.002
    namelist['damping']['Rayleigh']['z_d'] = 500.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['SMOKE']
    namelist['stats_io']['frequency'] = 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 3600.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'SMOKE'
    namelist['meta']['casename'] = 'SMOKE'

    return namelist

def Rico():

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 128
    namelist['grid']['ny'] = 128
    namelist['grid']['nz'] = 150
    namelist['grid']['gw'] = 7
    namelist['grid']['dx'] = 100.0
    namelist['grid']['dy'] = 100.0
    namelist['grid']['dz'] = 40.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0*24.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = False
    namelist['microphysics']['ccn'] = 70.0e6
    namelist['microphysics']['scheme'] = 'SB_Liquid'
    namelist['microphysics']['SB_Liquid'] = {}

    namelist['microphysics']['SB_Liquid']['nu_droplet'] = 0
    namelist['microphysics']['SB_Liquid']['mu_rain'] = 1

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = True
    namelist['tracers']['scheme'] = 'UpdraftTracers'
    namelist['tracers']['use_lcl_tracers'] = False
    namelist['tracers']['timescale'] = 15.0

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7
    namelist['scalar_transport']['order_sedimentation'] = 1

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 800

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Cumulus']
    namelist['stats_io']['frequency'] = 100.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy_frequency','viscosity']

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'Rico'
    namelist['meta']['casename'] = 'Rico'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    return namelist




def CGILS_S6(is_p2,is_ctl_omega):

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 96
    namelist['grid']['ny'] = 96
    namelist['grid']['nz'] = 180
    namelist['grid']['gw'] = 4
    namelist['grid']['dx'] = 100.0
    namelist['grid']['dy'] = 100.0
    namelist['grid']['dz'] = 30.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0*24.0*10.0 # 10 days

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'variable'


    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.02
    namelist['damping']['Rayleigh']['z_d'] = 600.0


    namelist['microphysics'] = {}
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = True
    namelist['microphysics']['ccn'] = 100.0e6
    namelist['microphysics']['scheme'] = 'SB_Liquid'
    namelist['microphysics']['SB_Liquid'] = {}

    namelist['microphysics']['SB_Liquid']['nu_droplet'] = 0
    namelist['microphysics']['SB_Liquid']['mu_rain'] = 1

    namelist['radiation'] = {}
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 90.0



    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] ={}
    namelist['sgs']['Smagorinsky']['iles'] = False

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7
    namelist['scalar_transport']['order_sedimentation'] = 1

    namelist['radiation'] = {}
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 90.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Cumulus']
    namelist['stats_io']['frequency'] = 5 * 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 86400.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy']

    namelist['meta'] = {}
    namelist['meta']['CGILS'] = {}
    namelist['meta']['casename'] = 'CGILS'
    namelist['meta']['CGILS']['location'] = 6
    namelist['meta']['CGILS']['P2'] = is_p2
    namelist['meta']['CGILS']['CTL_omega'] = is_ctl_omega

    simname = 'CGILS_S' + str(namelist['meta']['CGILS']['location'] )
    if namelist['meta']['CGILS']['P2']:
        if namelist['meta']['CGILS']['CTL_omega']:
            simname += '_P2'
        else:
            simname += '_P2S'
    else:
        simname += '_CTL'
    namelist['meta']['simname'] = simname



    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0
    namelist['restart']['delete_old'] = True
    namelist['restart']['times_retained'] = range(86400, 86400*11, 86400)

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 43200.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    return namelist







def CGILS_S11(is_p2,is_ctl_omega):

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 96
    namelist['grid']['ny'] = 96
    namelist['grid']['nz'] = 180
    namelist['grid']['gw'] = 4
    namelist['grid']['dx'] = 50.0
    namelist['grid']['dy'] = 50.0
    namelist['grid']['dz'] = 20.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0*24.0*10.0 # 10 days

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'variable'

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.02
    namelist['damping']['Rayleigh']['z_d'] = 600.0


    namelist['microphysics'] = {}
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = True
    namelist['microphysics']['ccn'] = 100.0e6
    namelist['microphysics']['scheme'] = 'SB_Liquid'
    namelist['microphysics']['SB_Liquid'] = {}

    namelist['microphysics']['SB_Liquid']['nu_droplet'] = 0
    namelist['microphysics']['SB_Liquid']['mu_rain'] = 1



    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] ={}
    namelist['sgs']['Smagorinsky']['iles'] = False

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7
    namelist['scalar_transport']['order_sedimentation'] = 1

    namelist['radiation'] = {}
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 90.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Flux']
    namelist['stats_io']['frequency'] = 5 * 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 86400.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy']

    namelist['meta'] = {}
    namelist['meta']['CGILS'] = {}
    namelist['meta']['casename'] = 'CGILS'
    namelist['meta']['CGILS']['location'] = 11
    namelist['meta']['CGILS']['P2'] = is_p2
    namelist['meta']['CGILS']['CTL_omega'] = is_ctl_omega

    simname = 'CGILS_S' + str(namelist['meta']['CGILS']['location'] )
    if namelist['meta']['CGILS']['P2']:
        if namelist['meta']['CGILS']['CTL_omega']:
            simname += '_P2'
        else:
            simname += '_P2S'
    else:
        simname += '_CTL'
    namelist['meta']['simname'] = simname



    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0
    namelist['restart']['delete_old'] = True
    namelist['restart']['times_retained'] = range(86400, 86400*11, 86400)

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 43200.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    return namelist





def CGILS_S12(is_p2,is_ctl_omega):

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 96
    namelist['grid']['ny'] = 96
    namelist['grid']['nz'] = 200
    namelist['grid']['gw'] = 4
    namelist['grid']['dx'] = 25.0
    namelist['grid']['dy'] = 25.0
    namelist['grid']['dz'] = 10.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0*24.0*10.0 # 10 days

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'variable'


    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.02
    namelist['damping']['Rayleigh']['z_d'] = 500.0

    namelist['microphysics'] = {}
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = True
    namelist['microphysics']['ccn'] = 100.0e6
    namelist['microphysics']['scheme'] = 'SB_Liquid'
    namelist['microphysics']['SB_Liquid'] = {}

    namelist['microphysics']['SB_Liquid']['nu_droplet'] = 0
    namelist['microphysics']['SB_Liquid']['mu_rain'] = 1



    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] ={}
    namelist['sgs']['Smagorinsky']['iles'] = False

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 7

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 7
    namelist['scalar_transport']['order_sedimentation'] = 1

    namelist['radiation'] = {}
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 90.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Flux']
    namelist['stats_io']['frequency'] = 5 * 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 86400.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy']

    namelist['meta'] = {}
    namelist['meta']['CGILS'] = {}
    namelist['meta']['casename'] = 'CGILS'
    namelist['meta']['CGILS']['location'] = 12
    namelist['meta']['CGILS']['P2'] = is_p2
    namelist['meta']['CGILS']['CTL_omega'] = is_ctl_omega

    simname = 'CGILS_S' + str(namelist['meta']['CGILS']['location'] )
    if namelist['meta']['CGILS']['P2']:
        if namelist['meta']['CGILS']['CTL_omega']:
            simname += '_P2'
        else:
            simname += '_P2S'
    else:
        simname += '_CTL'
    namelist['meta']['simname'] = simname



    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0
    namelist['restart']['delete_old'] = True
    namelist['restart']['times_retained'] = range(86400, 86400*11, 86400)

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 43200.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    return namelist




def ZGILS(zgils_loc):

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 86
    namelist['grid']['ny'] = 86
    namelist['grid']['nz'] = 216
    namelist['grid']['gw'] = 4
    namelist['grid']['dx'] = 75.0
    namelist['grid']['dy'] = 75.0
    namelist['grid']['dz'] = 20.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0*24.0*20.0 # 20 days

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'variable'


    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 500.0

    namelist['microphysics'] = {}
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'
    namelist['microphysics']['cloud_sedimentation'] = True
    namelist['microphysics']['ccn'] = 100.0e6
    namelist['microphysics']['scheme'] = 'SB_Liquid'
    namelist['microphysics']['SB_Liquid'] = {}
    namelist['microphysics']['SB_Liquid']['nu_droplet'] = 0
    namelist['microphysics']['SB_Liquid']['mu_rain'] = 1



    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    namelist['sgs']['Smagorinsky'] ={}
    namelist['sgs']['Smagorinsky']['iles'] = False


    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5
    namelist['scalar_transport']['order_sedimentation'] = 1

    namelist['surface_budget'] = {}
    if zgils_loc == 12:
        namelist['surface_budget']['ocean_heat_flux'] = 70.0
    elif zgils_loc == 11:
        namelist['surface_budget']['ocean_heat_flux'] = 90.0
    elif zgils_loc == 6:
        namelist['surface_budget']['ocean_heat_flux'] = 60.0

    # To run a fixed_sst case set fixed_sst_time > t_max of simulation
    namelist['surface_budget']['fixed_sst_time'] = 24.0 * 3600.0 * 30.0 # 3 days spinup

    namelist['radiation'] = {}
    namelist['radiation']['RRTM'] = {}
    namelist['radiation']['RRTM']['frequency'] = 90.0

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Flux']
    namelist['stats_io']['frequency'] = 5 * 60.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 86400.0
    namelist['fields_io']['diagnostic_fields'] = ['ql','temperature','buoyancy']

    namelist['meta'] = {}
    namelist['meta']['ZGILS'] = {}
    namelist['meta']['casename'] = 'ZGILS'
    namelist['meta']['ZGILS']['location'] = zgils_loc


    simname = 'ZGILS_S' + str(namelist['meta']['ZGILS']['location'] )
    namelist['meta']['simname'] = simname



    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0
    namelist['restart']['delete_old'] = True
    namelist['restart']['times_retained'] = range(86400, 86400*21, 86400)

    namelist['conditional_stats'] ={}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 43200.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'


    return namelist

def TRMM_LBA():
    # adopted from: "Daytime convective development over land- A model intercomparison based on LBA observations",
    # By Grabowski et al (2004)  Q. J. R. Meteorol. Soc. 132 317-344
    # modifications: we use less random perturbations than the original paper, our simulation does not change domain size in time
    # and has higher resolution in the BL when deep convection takes place

    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 200
    namelist['grid']['ny'] = 200
    namelist['grid']['nz'] = 220
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 100.0
    namelist['grid']['dy'] = 100.0
    namelist['grid']['dz'] = 100.0
    namelist['grid']['stretch'] = True

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0 * 6.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'Arctic_1M' #T_Liquid'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only' # liquid_ice

    #namelist['microphysics']['cloud_sedimentation'] = False
    #namelist['microphysics']['ccn'] = 70.0e6
    #namelist['microphysics']['SB_Liquid']['nu_droplet'] = 0
    #namelist['microphysics']['SB_Liquid']['mu_rain'] = 1

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = True
    namelist['tracers']['scheme'] = 'UpdraftTracers'
    namelist['tracers']['use_lcl_tracers'] = False
    namelist['tracers']['timescale'] = 60.0

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5 # the WENO order of the momentum advection scheme

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5 # the WENO order of the scaler advection scheme
    namelist['scalar_transport']['order_sedimentation'] = 1

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 4000

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Cumulus']
    namelist['stats_io']['frequency'] = 100.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['ql', 'temperature', 'buoyancy_frequency', 'viscosity', 'buoyancy' , 'thetali']

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'TRMM_LBA'
    namelist['meta']['casename'] = 'TRMM_LBA'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['conditional_stats'] = {}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    return namelist


def ARM_SGP():
    # adopted from: "Large-eddy simulation of the diurnal cycle of shallow cumulus convection over land",
    # By Brown et al. (2002)  Q. J. R. Meteorol. Soc. 128, 1075-1093
    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 8# 100
    namelist['grid']['ny'] = 8# 100
    namelist['grid']['nz'] = 110
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 64.0
    namelist['grid']['dy'] = 64.0
    namelist['grid']['dz'] = 40.0

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1 #6
    namelist['mpi']['nprocy'] = 1 #4
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 10.0
    namelist['time_stepping']['t_max'] = 3600.0 * 14.5

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'None_SA'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'

    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = True
    namelist['tracers']['scheme'] = 'UpdraftTracers'
    namelist['tracers']['use_lcl_tracers'] = False
    namelist['tracers']['timescale'] = 15.0

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5 # the WENO order of the momentum advection scheme

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5 # the WENO order of the scaler advection scheme
    namelist['scalar_transport']['order_sedimentation'] = 1

    namelist['damping'] = {} # I used the Bomex values here
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 900

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Cumulus']
    namelist['stats_io']['frequency'] = 100.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['ql', 'temperature', 'buoyancy_frequency', 'viscosity']

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'ARM_SGP'
    namelist['meta']['casename'] = 'ARM_SGP'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['conditional_stats'] = {}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    return namelist


def GATE_III():
    # adopted from: "Large eddy simulation of Maritime Deep Tropical Convection",
    # By Khairoutdinov et al (2009)  JAMES, vol. 1, article #15
    namelist = {}

    namelist['grid'] = {}
    namelist['grid']['dims'] = 3
    namelist['grid']['nx'] = 2048
    namelist['grid']['ny'] = 2048
    namelist['grid']['nz'] = 220
    namelist['grid']['gw'] = 3
    namelist['grid']['dx'] = 100.0
    namelist['grid']['dy'] = 100.0
    namelist['grid']['dz'] = 100.0
    namelist['grid']['stretch'] = True

    namelist['mpi'] = {}
    namelist['mpi']['nprocx'] = 1
    namelist['mpi']['nprocy'] = 1
    namelist['mpi']['nprocz'] = 1

    namelist['time_stepping'] = {}
    namelist['time_stepping']['ts_type'] = 3
    namelist['time_stepping']['cfl_limit'] = 0.7
    namelist['time_stepping']['dt_initial'] = 1.0
    namelist['time_stepping']['dt_max'] = 2.0
    namelist['time_stepping']['t_max'] = 3600.0 * 24.0

    namelist['thermodynamics'] = {}
    namelist['thermodynamics']['latentheat'] = 'constant'

    namelist['microphysics'] = {}
    namelist['microphysics']['scheme'] = 'Arctic_1M'  # T_Liquid'
    namelist['microphysics']['phase_partitioning'] = 'liquid_only'  # liquid_ice

    namelist['sgs'] = {}
    namelist['sgs']['scheme'] = 'Smagorinsky'
    # yair - add tracer transport to the simulation
    namelist['tracers'] = {}
    namelist['tracers']['use_tracers'] = True
    namelist['tracers']['scheme'] = 'UpdraftTracers'
    namelist['tracers']['use_lcl_tracers'] = False
    namelist['tracers']['timescale'] = 60.0

    namelist['diffusion'] = {}
    namelist['diffusion']['qt_entropy_source'] = False

    namelist['momentum_transport'] = {}
    namelist['momentum_transport']['order'] = 5 # the WENO order of the momentum advection scheme

    namelist['scalar_transport'] = {}
    namelist['scalar_transport']['order'] = 5 # the WENO order of the scaler advection scheme
    namelist['scalar_transport']['order_sedimentation'] = 1

    namelist['damping'] = {}
    namelist['damping']['scheme'] = 'Rayleigh'
    namelist['damping']['Rayleigh'] = {}
    namelist['damping']['Rayleigh']['gamma_r'] = 0.2
    namelist['damping']['Rayleigh']['z_d'] = 7000

    namelist['output'] = {}
    namelist['output']['output_root'] = './'

    namelist['stats_io'] = {}
    namelist['stats_io']['stats_dir'] = 'stats'
    namelist['stats_io']['auxiliary'] = ['Cumulus']
    namelist['stats_io']['frequency'] = 100.0

    namelist['fields_io'] = {}
    namelist['fields_io']['fields_dir'] = 'fields'
    namelist['fields_io']['frequency'] = 1800.0
    namelist['fields_io']['diagnostic_fields'] = ['ql', 'temperature', 'buoyancy_frequency', 'viscosity']

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'GATE_III'
    namelist['meta']['casename'] = 'GATE_III'

    namelist['restart'] = {}
    namelist['restart']['output'] = True
    namelist['restart']['init_from'] = False
    namelist['restart']['input_path'] = './'
    namelist['restart']['frequency'] = 600.0

    namelist['conditional_stats'] = {}
    namelist['conditional_stats']['classes'] = ['Spectra']
    namelist['conditional_stats']['frequency'] = 600.0
    namelist['conditional_stats']['stats_dir'] = 'cond_stats'

    return namelist



def write_file(namelist):

    try:
        type(namelist['meta']['simname'])
    except:
        print('Casename not specified in namelist dictionary!')
        print('FatalError')
        exit()

    namelist['meta']['uuid'] = str(uuid.uuid4())

    fh = open(namelist['meta']['simname'] + '.in', 'w')
    pprint.pprint(namelist)
    json.dump(namelist, fh, sort_keys=True, indent=4)
    fh.close()

    return


if __name__ == '__main__':
    main()
