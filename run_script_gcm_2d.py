import subprocess
import generate_namelist
import json 
import numpy as np
import netCDF4 as nc 
from collections import namedtuple 

lons = np.linspace(0,180,36) 
lons =  lons[:] 
#lons = lons[:19]  
#lons = [92] 
#import sys; sys.exit() 

times_retained = list(np.arange(100)* 86400)

pt_tup = namedtuple('pt_tup', ['lat', 'lon', 'lat_short', 'lon_short', 'name']) 

def main():

    file = '/cluster/scratch/presselk/forcing/new_1.00x_default.nc'
    rt_grp = nc.Dataset(file, 'r') 
    lats = rt_grp['lats'][:]
    lons = rt_grp['lons'][:] 
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

        nml['mpi']['nprocx'] = 1 
        nml['mpi']['nprocy'] = 24

        nml['damping']['Rayleigh']['z_d'] = 10000.0


   
        nml['grid']['dx'] = 1000.0 
        nml['grid']['dy'] = 1000.0 
        nml['grid']['gw'] = 3 

        nml['grid']['nx'] = 3 
        nml['grid']['ny'] = 64 
        nml['grid']['stretch'] = True

        nml['stats_io']['segment'] = True  
        
        nml['sgs']['Smagorinsky']['iles'] = True 
        
#        nml['gcm']['file'] = '/cluster/scratch/presselk//long_forcing/1_00x/f_data_tv_154.pkl'        
        nml['lw_tau0_eqtr'] = 7.2 
        nml['lw_tau0_pole'] = 1.8 
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
        nml['output']['output_root'] = '/cluster/scratch/presselk/2dcrm/refac/rk3_s_subs_new_pdv_gbp_2d/'
        #import sys; sys.exit() 
        generate_namelist.write_file(nml, new_uuid=False)
        run_str = 'bsub -G es_tapio -W 24:00 -n 24 mpirun python main.py ' + \
            nml['meta']['simname'] + '.in'
        print(run_str)
        subprocess.call([run_str], shell=True)

    return

if __name__ == "__main__":
    main()
