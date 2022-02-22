import numpy as np
import cPickle as pkl
import os
import netCDF4 as nc
from scipy import interpolate

def main(path, o_path):
    restart_dbase = parse_files(path, o_path)
    build_scratch_files(o_path, restart_dbase, quad=False)

    #print restart_dbase
    return


def build_scratch_files(o_path, restart_dbase, quad = False, vert_refine = False):

    for v in restart_dbase['PV_names']:
        print v 
        rt_grp = nc.Dataset(os.path.join(o_path, v + '_scratch.nc'), 'w', format='NETCDF4')

        if quad:
            quad_factor = 2.0
        else:
            quad_factor = 1.0

        #Create dimensions
        rt_grp.createDimension('x', restart_dbase['n'][0] * quad_factor + 2)
        rt_grp.createDimension('y', restart_dbase['n'][1] * quad_factor + 2)
        rt_grp.createDimension('z', restart_dbase['n'][2])


        rt_grp.createDimension('xh', restart_dbase['n'][0] * quad_factor + 2)
        rt_grp.createDimension('yh', restart_dbase['n'][1] * quad_factor + 2)
        rt_grp.createDimension('zh', restart_dbase['n'][2])

        #Create variables
        xh = rt_grp.createVariable('xh', 'f8', ('xh'))
        yh = rt_grp.createVariable('yh', 'f8', ('yh'))
        zh = rt_grp.createVariable('zh', 'f8', ('zh'))

        x = rt_grp.createVariable('x', 'f8', ('x'))
        y = rt_grp.createVariable('y', 'f8', ('y'))
        z = rt_grp.createVariable('z', 'f8', ('z'))

        #Write the physical coordinates
        x[:] = (-1.0 + np.arange((restart_dbase['n'][0]*quad_factor+2), dtype=np.double) )* restart_dbase['dx'][0]
        y[:] = (-1.0 + np.arange((restart_dbase['n'][1]*quad_factor+2), dtype=np.double) )* restart_dbase['dx'][1]
        z[:] = np.arange(restart_dbase['n'][2], dtype=np.double)*restart_dbase['dx'][2] + restart_dbase['dx'][2]

        xh[:] = (-1.0 + np.arange((restart_dbase['n'][0]*quad_factor+2), dtype=np.double) )* restart_dbase['dx'][0] + restart_dbase['dx'][0]/2.0
        yh[:] = (-1.0 + np.arange((restart_dbase['n'][1]*quad_factor+2), dtype=np.double) )* restart_dbase['dx'][1] + restart_dbase['dx'][1]/2.0
        zh[:] = np.arange(restart_dbase['n'][2], dtype=np.double)*restart_dbase['dx'][2] + restart_dbase['dx'][2]/2


        #Setup the data array
        data = rt_grp.createVariable('data', 'f8', ('x', 'y', 'z'))


        #Now loop over pickle files and build datasets
        for f in restart_dbase['restart_files']:
            #print v, f
            fh = open(os.path.join(restart_dbase['path'], f), 'rb')
            d = pkl.load(fh)
            start = restart_dbase[f][v + '_start']
            end = start +restart_dbase[f]['nl'][0] * restart_dbase[f]['nl'][1] * restart_dbase[f]['nl'][2]

            #print 'npg', restart_dbase[f]['nl'], np.shape(d['PV']['values'])[0] /5,start, end

            #print restart_dbase[f]['npg'], end - start

            #print 'nlg: ', restart_dbase[f]['nlg'], ' PV values shape: ', np.shape(d['PV']['values'])
            #print np.shape(d['PV']['values'][start:end]), restart_dbase[f]['nlg'][0] * restart_dbase[f]['nlg'][1] * restart_dbase[f]['nlg'][2]
            data_tmp = d['PV']['values'][start:end].reshape((restart_dbase[f]['nl'][0],restart_dbase[f]['nl'][1],restart_dbase[f]['nl'][2]))

            indx_lo = restart_dbase[f]['indx_lo'] + 1
            nl  = restart_dbase[f]['nl']
            n= restart_dbase['n']
            gw = restart_dbase['gw']

            #if v == 'w':
            #    data[indx_lo[0]:indx_lo[0] + nl[0],
           #     indx_lo[1]:indx_lo[1] + nl[1],
           #     :] = data_tmp * restart_dbase[
           #                                                                                            'rho'][
           #                                                                                        np.newaxis,
           #                                                                                        np.newaxis, gw:-gw]
           # else:
            data[indx_lo[0]:indx_lo[0] + nl[0], indx_lo[1]:indx_lo[1] + nl[1], :] = data_tmp #* restart_dbase['rho_half'][np.newaxis, np.newaxis,gw:-gw]


            rt_grp.sync()
            if quad:

                data[n[0] + indx_lo[0]:n[0] + indx_lo[0] + nl[0],
                     n[1] + indx_lo[1]:n[1] + indx_lo[1] + nl[1],
                     :]  = data_tmp
                rt_grp.sync()

                data[indx_lo[0]:indx_lo[0] + nl[0],
                     n[1] + indx_lo[1]:n[1] + indx_lo[1] + nl[1],
                     :]  = data_tmp
                rt_grp.sync()


                data[n[0] + indx_lo[0]:n[0] + indx_lo[0] + nl[0],
                     indx_lo[1]:indx_lo[1] + nl[1],
                     :]  = data_tmp
                rt_grp.sync()

            data[0,:,:] = data[-2,:,:]
            rt_grp.sync()
            data[-1,:,:] = data[1,:,:]
            rt_grp.sync()
            data[:,0,:] = data[:,-2,:]
            rt_grp.sync()
            data[:,-1,:] = data[:,1,:]
            rt_grp.sync()
            fh.close()




        rt_grp.close()

    return





def parse_files(path, o_path):
    
    restart_files = os.listdir(path)
    restart_dbase = {}
    #print restart_files, path 
    restart_dbase['path'] = path
    restart_dbase['restart_files'] = restart_files
    for f in restart_files:
        #print f
        #Create a dictionary for each file
        restart_dbase[f] = {}

        #Open file and begin getting data
        fh = open(os.path.join(path, f), 'rb')
        d = pkl.load(fh)


        #print d['Gr']
        if f == restart_files[0]:
            #Mapping of names to indicies
            restart_dbase['PV_names'] = d['PV']['index_name']
            restart_dbase['PV_index'] = d['PV']['name_index']
            restart_dbase['n'] = d['Gr']['n']

            restart_dbase['dx'] = d['Gr']['dx']
            restart_dbase['rho'] = 1.0/d['Ref']['alpha0']
            restart_dbase['rho_half'] = 1.0/d['Ref']['alpha0_half']
            restart_dbase['gw'] = d['Gr']['gw']
        #print restart_dbase['gw']
        #import sys; sys.exit()


        restart_dbase[f]['npl'] = d['Gr']['npl']
        restart_dbase[f]['nl'] = d['Gr']['nl']
        restart_dbase[f]['npg'] = np.product(d['Gr']['npg'])
        restart_dbase[f]['nlg'] = d['Gr']['nlg']
        restart_dbase[f]['indx_lo'] = d['Gr']['indx_lo']
        #print restart_dbase[f]['nl']

        #These are the starts for each variables
        for v in restart_dbase['PV_names']:
            restart_dbase[f][v + '_start'] = restart_dbase['PV_index'][v] * restart_dbase[f]['nl'][0]*  restart_dbase[f]['nl'][1] *  restart_dbase[f]['nl'][2]






        fh.close()

    #Using the last file restart dump a pickle with auxillary data.
    d.pop('Gr', None)
    d.pop('PV', None)
    #d.pop('Ref', None)


    #print d.keys()


    f_out = file(os.path.join(o_path, 'aux_data.pkl'),'wb')
    pkl.dump(d,f_out)
    f_out.close()




    return restart_dbase



if __name__ == '__main__':
    #o_path = '/cluster/scratch/presselk/2dcrm/refac/rk3_s_subs_new_pdv_gbp_sa_restart/Output.1_00x_140.625.1_00x_140.625/altered_fields/' 
    o_path = '/central/groups/esm/zhaoyi/GCMForcedLES/altered_fields/1_00x/250m/90.0/18280800/' 
    #path =  '/cluster/scratch/presselk/2dcrm/refac/rk3_s_subs_new_pdv_gbp_sa_restart/Output.1_00x_140.625.1_00x_140.625/Restart/10368000/'
    path = '/central/scratch/zhaoyi/1_00x/Output.1_00x_90.0_250m.4x/Restart/18280800/' 
    main(path, o_path)
