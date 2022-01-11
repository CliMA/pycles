import netCDF4 as nc
#import pylab as plt
import numpy as np
import os
from scipy.signal import savgol_filter
from scipy.interpolate import pchip
from scipy.integrate import cumtrapz 

def compute_derivative(xp, yp):


    dyp_dxp = np.diff(yp)/np.diff(xp)
    
    
    return dyp_dxp, 0.5 * (xp[1:] + xp[:-1])

def mdi_interp(x, y, xn): 
    xp = x.filled() 
    yp = y.filled()

    p_interp_val = pchip(xp, yp)
    dyp_dxp, dir_loc = compute_derivative(xp, yp)
    p_interp1= pchip(dir_loc, dyp_dxp)

    cumtrapz(p_interp1(xn), xn, initial= 0.0)
    #return cumtrapz(np.interp(xn, dir_loc, dyp_dxp)[::--1, xn, initial= 0.0), xn
    #print np.array(cumtrapz(p_interp1(xn)[::-1], xn[::-1], initial= 0.0)[::-1] + p_interp_val(np.max(xn)))
    return np.array(cumtrapz(p_interp1(xn)[::-1], xn[::-1], initial= 0.0)[::-1] + p_interp_val(np.max(xn))), xn



class cfreader_grid:

    def __init__(self, path, lat, lon):

        self.file = path
        self.lat = lat
        self.lon = lon

        return

    def get_profile_mean(self, var, zero_bottom=False):

        op_grp = nc.Dataset(self.file, 'r')
        ntime = np.size(op_grp.dimensions['time'])
        nlev = np.size(op_grp.dimensions['lev'])
        lat = np.array(op_grp.variables['lat'])
        lon = np.array(op_grp.variables['lon'])
        lat_mask = np.round(lat)==np.round(self.lat)
        lon_mask = np.round(lon)==np.round(self.lon)
        var_handle = op_grp.variables[var][...,lat_mask,lon_mask].squeeze()
        assert ((ntime, nlev) == var_handle.shape or (ntime,) == var_handle.shape)
        data = np.mean(var_handle, axis=0)
        op_grp.close()
        #print var, data

        if zero_bottom:
            return np.append(data, 0.0)[:-1]
        else:
            return np.append(data, data[-1])[:-1]

    def get_timeseries_mean(self, var):

        op_grp = nc.Dataset(self.file, 'r')
        ntime = np.size(op_grp.dimensions['time'])
        nlev = np.size(op_grp.dimensions['lev'])
        lat = np.array(op_grp.variables['lat'])
        lon = np.array(op_grp.variables['lon'])
        lat_mask = np.round(lat)==np.round(self.lat)
        lon_mask = np.round(lon)==np.round(self.lon)
        var_handle = op_grp.variables[var][...,lat_mask,lon_mask].squeeze()
        assert((ntime,) == var_handle.shape)
        data = var_handle[:]
        op_grp.close()

        return np.mean(data, axis=0)

    def get_interp_profile(self, var, z, zero_bottom=False, filter=True):
        
        '''
        :param var: name of variable in fms data
        :param z: interpolation points
        :param zero_bottom: bool to specify if bottom boundary is set to zero if not then take value of one point above
        :return: array of var interpolated onto z
        '''

        data = self.get_profile_mean(var, zero_bottom)
        z_gcm = self.get_profile_mean('zg', zero_bottom=True)

        yn, dir_loc = mdi_interp(z_gcm, data, z) 

        return yn 

    def get_interp_profile_old(self, var, z, zero_bottom=False, filter=False):
        
        '''
        :param var: name of variable in fms data
        :param z: interpolation points
        :param zero_bottom: bool to specify if bottom boundary is set to zero if not then take value of one point above
        :return: array of var interpolated onto z
        '''

        data = self.get_profile_mean(var, zero_bottom)
        z_gcm = self.get_profile_mean('zg', zero_bottom=True)
        #print data 
        #print z_gcm 
        p_interp1= pchip(z_gcm[:].filled(), data[:].filled())#np.interp(z, z_gcm[:], data[:])
        z_interp1 = np.linspace(0.0, np.max(z_gcm), 2560)
        data_interp1 = p_interp1(z_interp1)

        #plt.figure()
        #plt.plot(data_interp1, z_interp1)

        if not filter:
            return p_interp1(z)
        if filter:
            data_interp1 = savgol_filter(data_interp1, 37, 3)


        p_interp2 = pchip(z_interp1.filled(), data_interp1.filled())
        return p_interp2(z)


        #plt.plot(data_interp1, z_interp1)#
        #
        #plt.show()
        #
        #import sys; sys.exit()

        #f not filter:
        #    return pinter
        #else:
        #    return savgol_filter(data_interp, 37, 3)


    def get_value(self, var):
        rt_grp = nc.Dataset(self.file, 'r')
        op_grp = rt_grp[self.op_grp]
        var_handle =  op_grp.variables[var]
        assert(() == var_handle.shape)

        return var_handle[0]


def main():

    return

if __name__ == "__main__":
    path = './AM4_amip.2009-2014.07.nc'
    lat = 17.0
    lon = 211.25

    rdr = cfreader(path, lat, lon)

    #t = rdr.get_profile_mean('temp')
    #sphum = rdr.get_profile_mean('sphum')
    #ucomp = rdr.get_profile_mean('ucomp')
    #vcomp = rdr.get_profile_mean('vcomp')
    #alpha = rdr.get_profile_mean('alpha')

    #Test interpolation
    interp_test_dir = './InterpTests/'
    if not os.path.exists(interp_test_dir):
        os.makedirs(interp_test_dir )

    vars = ['ta', 'hus',
            'ua', 'va']
    height_gcm = rdr.get_profile_mean('zg')
    height_les = np.linspace(0.0, 25600.0, 256)
    for v in vars:
        var_gcm = rdr.get_profile_mean(v)
        var_les_filt = rdr.get_interp_profile(v, height_les)
        var_les = rdr.get_interp_profile(v, height_les, filter=False)

        plt.figure()
        plt.plot(var_gcm, height_gcm, 'o')
        plt.plot(var_les, height_les)
        plt.plot(var_les_filt, height_les, '.')
        plt.savefig(os.path.join(interp_test_dir, v + '_linear.pdf'))
        plt.close()

    ts_mean = rdr.get_timeseries_mean('ts')
    print(ts_mean)

    main()
