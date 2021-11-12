import subprocess
import generate_namelist
import json 
import numpy as np
import netCDF4 as nc 
from collections import namedtuple 
import numpy as np

lons = np.linspace(0,180,36) 
lons =  lons[:] 
#lons = lons[:19]  
#lons = [92] 
#import sys; sys.exit() 

times_retained = list(np.arange(120)* 86400)

pt_tup = namedtuple('pt_tup', ['lat', 'lon', 'lat_short', 'lon_short', 'name']) 

tau_factor = 1.0 
def main():

    file = '/home/kpressel/FixedForcingPaper/2dcrm/forcing_data/new_1.20x_default.nc'
    rt_grp = nc.Dataset(file, 'r') 
    lats = rt_grp['lats'][::4]
    lons = rt_grp['lons'][::4] 
    assert(len(lats) == len(lons))
    pts = [] 
    for i in range(len(lats)): 
        lat_short = np.round(lats[i], 1)
        lon_short = np.round(lons[i], 1)
        pts.append(pt_tup(lat=lats[i], lon=lons[i], lat_short = lat_short, lon_short=lon_short, 
            name= str(lat_short) + '_' + str(lon_short)))
        #print pts 
    
    pts = pts[::-1] 

    for pt in pts:
        #nml = generate_namelist.StableBubble()
        file_namelist = open('GCMVarying.in', 'r').read()  
        nml = json.loads(file_namelist)


        nml['meta']['simname'] = '1_00x_' + str(pt.lon)
        nml['meta']['uuid'] = nml['meta']['simname'] 
#       nml['scalar_transport']['order'] = scheme
#        nml['momentum_transport']['order'] = scheme

        nml['mpi']['nprocx'] = 12
        nml['mpi']['nprocy'] =  6

        nml['damping']['Rayleigh']['z_d'] = 10000.0


   
        nml['grid']['dx'] = 1000.0 
        nml['grid']['dy'] = 1000.0 
        nml['grid']['gw'] = 3 

        nml['grid']['nx'] = 64 
        nml['grid']['ny'] = 64 
        nml['grid']['stretch'] = True

        nml['stats_io']['segment'] = True  
        
        nml['sgs']['Smagorinsky']['iles'] = False 
        
#        nml['gcm']['file'] = '/cluster/scratch/presselk//long_forcing/1_00x/f_data_tv_154.pkl'        
        nml['gcm']['lw_tau0_eqtr'] = 7.2 * tau_factor 
        nml['gcm']['lw_tau0_pole'] = 1.8 * tau_factor
        nml['restart']['times_retained'] = times_retained 

        nml['gcm']['file'] = file
        nml['gcm']['lat'] =  pt.lat
        nml['gcm']['lon'] =  pt.lon 

        nml['surface_budget'] = {} 
        nml['surface_budget']['water_depth_initial'] = 1.0  
        nml['surface_budget']['water_depth_final'] = 1.0 
        nml['surface_budget']['fixed_sst_time'] = 24.0 * 3600.0 
        print nml['gcm']['file'] 
        nml['time_stepping']['dt_max'] = 15.0 
        nml['time_stepping']['cfl_limit'] = 0.7 
        nml['time_stepping']['t_max'] = 86400.0 * 120.0
        nml['time_stepping']['acceleration_factor'] = 8.0 
        nml['time_stepping']['ts_type'] = 3 
        nml['visualization'] = {} 
        nml['visualization']['frequency'] = 180000.0
        nml['output']['output_root'] = '/central/scratch/kpressel/1.20x/'
        #import sys; sys.exit() 
        generate_namelist.write_file(nml, new_uuid=False)


        slurm_str = '''#!/bin/bash
 
#Submit this script with: sbatch run_center.sh
#SBATCH --time=96:00:00   # walltime
#SBATCH --ntasks=72   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "bomex_test"   # job name
#SBATCH --mail-user=kyle@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
 
 
# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
'''

        slurm_str += 'srun python main.py ' + nml['meta']['simname'] + '.in'

        f = open('run_'+ nml['meta']['simname'] + '.sh', 'w') 
        f.write(slurm_str) 
        f.close() 

        #run_str = 'bsub -G es_tapio -W 120:00 -n 72 mpirun python main.py ' + \
        #    nml['meta']['simname'] + '.in'
        run_str = 'sbatch ' + 'run_'+ nml['meta']['simname'] + '.sh' 
        print(run_str)
        subprocess.call([run_str], shell=True)

    return

if __name__ == "__main__":
    main()
