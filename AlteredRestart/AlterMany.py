import AlteredRestart
import glob 
import os 
import netCDF4 as nc 
import numpy as np
import subprocess 

def main():
        
    tau0 = '1_00x'
    ars = {} 
    ars['output_altered_root'] = '/central/groups/esm/zhaoyi/GCMForcedLES/altered_fields/'+tau0+'/250m/'
    ars['output_restart_root'] = '/central/groups/esm/zhaoyi/GCMForcedLES/restart/'+tau0+'/250m/'
    ars['restart_root'] = '/central/scratch/zhaoyi/'+tau0+'/'
    ars['restart_time'] = '20736000'
    
    filename = '/central/groups/esm/zhaoyi/GCMForcedLES/forcing/new_'+tau0+'_default.nc'
    rt_grp = nc.Dataset(filename, 'r')
    lons = rt_grp['lons'][0::4]
    rt_grp.close() 
     
    ars['case_list'] = list(np.round(lons,1)[:4])
    print ars['case_list'] 
            
    make_paths(ars) 
    get_restarts(ars)
    
    
    for case in ars['case_restarts'].keys(): 
        out_altered_path = ars['case_altered_paths'][case]
        out_restart_path = ars['case_restart_paths'][case]
        restart_path = ars['case_restarts'][case] 
        
        print case, out_altered_path, out_restart_path, restart_path 
        returned_value = os.system('cp -r '+restart_path+' '+out_restart_path)
        print "cp returned value: ", returned_value
        restart_dbase = AlteredRestart.parse_files(restart_path, out_altered_path)
        AlteredRestart.build_scratch_files(out_altered_path, restart_dbase, quad=False)
 
    return 

def make_paths(ars):

    #Generate Altered Restarts 
    if not os.path.exists(ars['output_altered_root']):
        os.mkdir(ars['output_altered_root'])
    if not os.path.exists(ars['output_restart_root']):
        os.mkdir(ars['output_restart_root'])

    ars['case_altered_paths'] = {}
    ars['case_restart_paths'] = {}     
    #Now create the paths for each of the individual cases 
    for case in ars['case_list']:
    
        path = os.path.join(ars['output_altered_root'], str(case))
        path_time = os.path.join(path, ars['restart_time'])
        ars['case_altered_paths'][case] = path_time 
        if not os.path.exists(path):
            os.mkdir(path) 
        if not os.path.exists(path_time):
            os.mkdir(path_time) 
        path = os.path.join(ars['output_restart_root'], str(case))
        ars['case_restart_paths'][case] = path 
        if not os.path.exists(path):
            os.mkdir(path) 
    
    return
    
def get_restarts(ars): 

    sim_dirs = glob.glob(os.path.join(ars['restart_root'], 'Output*250m.4x')) 
    
    ars['case_restarts'] = {} 
    for case in ars['case_list']: 
        for sim in sim_dirs: 
            if str(case) in sim: 
                ars['case_restarts'][case] = os.path.join(os.path.join(sim, 'Restart'), ars['restart_time'])
                
                

    return  


if __name__ == '__main__': 
    main() 

