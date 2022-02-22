import glob 
import json 
import os 
import subprocess 
import netCDF4 as nc
import numpy as np
def main(cases):

    for case in cases: 
        f = glob.glob('./*' + case + '*.restart_4x')[0]  
       	nml_out = None
        with open(f, 'r') as fh: 
            nml_out = json.load(fh)  
       

	#Now get the latest restart file
        #print nml_out['output']['output_root']
        #print case 
        out_dir = glob.glob(os.path.join(nml_out['output']['output_root'], '*' + case + '*' ))[0]
        print out_dir
        restart_path = os.path.join(out_dir, 'Restart') 
        
        #print restart_path 
        last_restart=None
        try: 
            last_restart = glob.glob(os.path.join(restart_path, '12960000'))[0]
            #print ' contins ', os.listdir(restart_path)[:]
            #print last_restart
        except:
            print restart_path  
            last_restart = glob.glob(os.path.join(restart_path, '*new*'))[0]
 



        print last_restart


        #print sorted(glob.glob(os.path.join(restart_path, '*'))[:]) 
        #import sys; sys.exit() 

        nml_out['restart']['init_altered']  = False
        nml_out['restart']['init_from'] = True 
        nml_out['restart']['input_path'] = last_restart 
        nml_out['restart']['frequency'] = 3600.0 
        nml_out['time_stepping']['cfl_limit'] = 0.7
        nml_out['time_stepping']['t_max'] = 11232000.0 + 40*86400.0 	
        f_out = f + '.restart_4x_restart'
        if last_restart is not None:
            with open(f_out, 'w') as fh:
                json.dump(nml_out, fh, sort_keys=True, indent=4)


            run_str = 'bsub -G es_tapio -W 120:00 -n 144 mpirun python main.py ' + f_out
            print(run_str) 
            subprocess.call([run_str], shell=True)

    return 



if __name__ == '__main__': 

    #cases = ['95.625', '98.4375', '101.25', '104.0625', '112.5', '115.3125', '118.125', '120.9375', 
    #         '123.75', '126.5625', '129.375', '132.1875'] 

    file = '/cluster/scratch/presselk/forcing/new_1.00x_default.nc'
    rt_grp = nc.Dataset(file, 'r')
    lats = rt_grp['lats'][::-4]
    lons = rt_grp['lons'][::-4]

    print lons 

    #lons = [171.5625, 168.75, 165.9375, 157.5, 154.6875, 151.875, 149.0625, 146.25, 143.4375] 
    #lats = [lats] * len(lons)   
    #lons = lons[::-1] 
    rt_grp.close() 
    cases = []
    for i in range(len(lats)):
         lat_short = np.round(lats[i], 1)
         lon_short = np.round(lons[i], 1)
         cases.append(str(lons[i])) 
    cases = ['90.0', '101.25', '112.5']#, '123.75', '135.0'] 
    #print cases
    main(cases) 



