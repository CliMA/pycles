import glob 
import os 
import netCDF4 as nc 
import numpy as np
import subprocess 

def main():
    import combine3dFields
        
    tau0 = '1_00x'
    ars = {} 
    ars['output_root'] = '/export/data1/zhaoyi/GCMForcedLES/fields_combined/'+tau0+'/250m/'
    ars['fields_root'] = '/export/data1/zhaoyi/GCMForcedLES/'+tau0+'/250m/'
    ars['restart_time'] = '20736000'
    
    filename = '/export/data1/zhaoyi/GCMForcedLES/forcing/new_'+tau0+'_default.nc'
    rt_grp = nc.Dataset(filename, 'r')
    lons = rt_grp['lons'][0::4]
    rt_grp.close() 
     
    ars['case_list'] = list(np.round(lons,1)[0:4])
    
            
    make_paths(ars) 
    get_fields(ars)
    
    
    for case in ars['case_fields'].keys(): 
        out_path = ars['case_fields_paths'][case]
        fields_path = ars['case_fields'][case] 
        
        print case, out_path, fields_path 
        combine3dFields.main(fields_path, out_path)
 
    return 

def make_paths(ars):

    #Generate Altered Restarts 
    if not os.path.exists(ars['output_root']):
        os.mkdir(ars['output_root'])
    
    ars['case_fields_paths'] = {}
    #Now create the paths for each of the individual cases 
    for case in ars['case_list']:
    
        path = os.path.join(ars['output_root'], str(case))
        path_time = os.path.join(path, ars['restart_time'])
        ars['case_fields_paths'][case] = path_time 
        if not os.path.exists(path):
            os.mkdir(path) 
        if not os.path.exists(path_time):
            os.mkdir(path_time) 
    
    return
    
def get_fields(ars): 

    sim_dirs = glob.glob(os.path.join(ars['fields_root'], 'Output*')) 
    
    ars['case_fields'] = {} 
    for case in ars['case_list']: 
        for sim in sim_dirs: 
            if str(case) in sim: 
                ars['case_fields'][case] = os.path.join(os.path.join(sim, 'fields'), ars['restart_time'])
                
                

    return  


if __name__ == '__main__': 
    main() 

