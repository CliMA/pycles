import pylab as plt
import numpy as np
import netCDF4 as nc
import cPickle as pkl
t_min = 1

#f = './Stats.GCMVarying.nc'
f = './Output.GCMVarying.9ddb9/stats/Stats.GCMVarying.nc'
f_pkl = './forcing/0_66x/f_data_tv_70.pkl'
f_pkl = file(f_pkl, 'r')
d_pkl = pkl.load(f_pkl)
f_pkl.close()

lat = 30.0
lat_in = d_pkl['lat']
lat_idx = (np.abs(lat_in - lat)).argmin()

print d_pkl.keys()

u_gcm  = d_pkl['u'][:,lat_idx]
v_gcm  = d_pkl['v'][:,lat_idx]
lw_flux_dn_gcm = d_pkl['lw_flux_dn'][:,lat_idx]
lw_flux_up_gcm = d_pkl['lw_flux_up'][:,lat_idx]
lwdn_gcm  = d_pkl['lwdn_sfc'][lat_idx]
lwup_gcm  = d_pkl['lwup_sfc'][lat_idx]
swdn_gcm  = d_pkl['swdn_sfc'][lat_idx]
swdn_toa_gcm =  d_pkl['swdn_toa'][lat_idx]
rad_heating_gcm = d_pkl['temp_rad'][:,lat_idx]
fino_heating_gcm = d_pkl['temp_fino'][:,lat_idx]
vadv_heating_gcm = d_pkl['temp_vadv'][:,lat_idx]
total_heating_gcm = d_pkl['temp_total'][:,lat_idx]
hadv_heating_gcm = d_pkl['temp_hadv'][:,lat_idx]
param_heating_gcm = d_pkl['temp_param'][:,lat_idx]
diffusion_heating_gcm = d_pkl['temp_diffusion'][:,lat_idx]
t_gcm = d_pkl['t'][:,lat_idx]
z_gcm = d_pkl['z'][:, lat_idx]
shf_gcm = d_pkl['shf_flux'][lat_idx]
lhf_gcm = d_pkl['lhf_flux'][lat_idx]


vadv_shum_gcm = d_pkl['shum_vadv'][:,lat_idx]
hadv_shum_gcm = d_pkl['shum_hadv'][:,lat_idx]

rt_grp = nc.Dataset(f,'r') 


profiles = rt_grp['profiles'] 
reference = rt_grp['reference'] 
timeseries = rt_grp['timeseries'] 
 
z = reference['zp_half'][:] 
sst = timeseries['surface_temperature'][:] 
t = timeseries['t'][:]/(3600.0*24.0)  
cloud_top = timeseries['cloud_top'][:] 

shf = timeseries['shf_surface_mean'][:]
lhf = timeseries['lhf_surface_mean'][:]
lwdn_sfc = timeseries['srf_lw_flux_down'][:]
swdn_sfc = timeseries['srf_sw_flux_down'][:]
lwup_sfc = timeseries['srf_lw_flux_up'][:]

u_mean = profiles['u_mean'][:,:].T
ql_mean = profiles['ql_mean'][:,:].T
qv_mean = profiles['qv_mean'][:,:].T
s_mean = profiles['s_mean'][:,:].T
cloud_fraction = profiles['cloud_fraction'][:,:].T
#qr_mean = profiles['qr_mean'][:,:].T
thetali_mean = profiles['thetali_mean'][:,:].T
temperature_mean = profiles['temperature_mean'][:,:].T
dsdt_hadv_mean = profiles['ls_dsdt_hadv'][:,:].T
sw_flux_down = profiles['sw_flux_down'][:,:].T
lw_flux_down = profiles['lw_flux_down'][:,:].T
lw_flux_up = profiles['lw_flux_up'][:,:].T
ls_subsidence = profiles['ls_subsidence'][:,:].T
t0 = reference['temperature0'][:]
ls_subs_dtdt= profiles['ls_subs_dtdt'][:,:].T
ls_subs_dsdt= profiles['ls_subs_dtdt'][:,:].T
ls_hadv_dtdt= profiles['ls_dtdt_hadv'][:,:].T
ls_fino_dtdt= profiles['ls_dtdt_fino'][:,:].T
#ls_eddy_dtdt= profiles['ls_dtdt_eddy'][:,:].T
ls_dqtdt_hadv = profiles['ls_dqtdt_hadv'][:,:].T
#ls_dqtdt_eddy = profiles['ls_dqtdt_eddy'][:,:].T
ls_subs_dqtdt = profiles['ls_subs_dqtdt'][:,:].T
grey_rad_dsdt = profiles['grey_rad_dsdt'][:,:].T
grey_rad_heating_mean = profiles['grey_rad_heating'][:,:].T

precip_dqtdt = profiles['dqtdt_precip_mean'][:,:].T



plt.close()
plt.figure()

z /= 1000.0
z_gcm /=1000.0
balance_time = -1
plt.plot(vadv_shum_gcm + hadv_shum_gcm , z_gcm, 'or',label='GCM NET', linewidth=4.0)
#plt.plot(ls_dqtdt_eddy[:,balance_time] + ls_subs_dqtdt[:,balance_time] + ls_dqtdt_hadv[:,balance_time], z , '-k', label='LES Net', linewidth=3.0, alpha=0.75)
#plt.plot(hadv_shum_gcm , z_gcm, '+r', label='hadv')
#plt.plot(vadv_shum_gcm , z_gcm, '*r', label='vadv')
plt.plot(ls_dqtdt_hadv[:,balance_time],z, '--k', label='LES hadv', linewidth=3.0)
plt.plot(ls_subs_dqtdt[:,balance_time],z, '-.k', label='LES subs', linewidth=3.0)
#plt.plot(ls_dqtdt_eddy[:,balance_time],z, ':k', label='LES eddy', linewidth=3.0)

plt.grid()
plt.ylim(0.0,25.6)
plt.xlabel('Specific Humidity Tendency kg/(kg s)')
plt.ylabel('height')
#plt.plot(ls_subs_dqtdt[:,0], '-k', label='subs')

plt.plot()
plt.legend(loc='upper left')
plt.savefig('balance_terms_qt.png')
plt.savefig('balance_terms_qt.pdf')
plt.close()

plt.figure()
plt.plot(lw_flux_dn_gcm, z_gcm)
plt.plot(lw_flux_down[:,1], z, '.r')
print lw_flux_up[0,1], lw_flux_up_gcm[-1]
plt.savefig('lw_flux_dn.png')
plt.savefig('lw_flux_dn.pdf')
plt.close()

plt.figure()
plt.plot(lw_flux_up_gcm, z_gcm)
plt.plot(lw_flux_up[:,1], z, '.r')
plt.savefig('lw_flux_up.png')
plt.savefig('lw_flux_up.pdf')
plt.close()

plt.figure()
plt.plot(u_mean[:,-1], z, '-b')
plt.plot(u_gcm, z_gcm, 'ob')
plt.savefig('u.png')
plt.close()


plt.figure()
plt.plot(t0, z, linewidth=2.5, label='T0 LES')
plt.plot(temperature_mean[:,0], z , linewidth=2.5, label='T LES')
plt.plot(t_gcm,z_gcm, 'or', label='T GCM')
plt.legend()
plt.ylabel('Height [m]')
plt.xlabel('Temperature [K]')
plt.savefig('t_comp.pdf')
#plt.show()
plt.close()


balance_time = -1
plt.figure()
plt.plot(vadv_heating_gcm + hadv_heating_gcm + fino_heating_gcm + rad_heating_gcm, z_gcm, 'or', label='GCM Net')
#plt.plot(ls_fino_dtdt[:,balance_time] + ls_eddy_dtdt[:,balance_time] + ls_hadv_dtdt[:,balance_time] + grey_rad_heating_mean[:,balance_time] + ls_subs_dtdt[:,balance_time],
#         z, '-k', label='LES Net', linewidth=3.0,alpha=0.75)
plt.plot(ls_fino_dtdt[:,balance_time], z, '-*k',linewidth=3.0, label='LES fino')
#plt.plot(ls_eddy_dtdt[:,1], z, ':k',linewidth=3.0, label='eddy')
#plt.plot(ls_subs_dtdt[:,balance_time] + ls_eddy_dtdt[:,balance_time], z, '-.k',linewidth=3.0, label=' LES vadv + eddy')
plt.plot(ls_subs_dtdt[:,balance_time] , z, '-.k',linewidth=3.0, label=' LES vadv')
plt.plot(ls_hadv_dtdt[:,balance_time], z, '--k',linewidth=3.0, label='LES hadv')
plt.plot(grey_rad_heating_mean[:,balance_time], z, 'b', label='LES Rad')



plt.legend(loc='upper left')
plt.xlabel('dT/dt')
plt.ylabel('z [m]')
plt.ylim(0.0,25.6)
plt.grid() 
plt.savefig('balance_terms.png')
plt.savefig('balance_terms.pdf')

print 'lat_idx', lat_idx

plt.figure()
plt.contour(precip_dqtdt,levels=[-0.0000001, 0.0])
plt.colorbar()
plt.savefig('precip_rate.png')
plt.close()



plt.figure()
plt.plot(rad_heating_gcm, z_gcm)
plt.plot(grey_rad_heating_mean[:,1], z, '.')
plt.savefig('rad_heating_rates')
plt.close()

print "Rad Heating", grey_rad_heating_mean[:,1]

qt_mean = profiles['qt_mean'][:,:].T
print "here"
plt.figure() 
plt.plot(t, sst)
plt.grid()  
plt.xlabel('Time [days]')
plt.ylabel('T [K]') 
plt.savefig('sst.png') 

print "here"
plt.figure()
plt.contourf(ql_mean, 200)
plt.colorbar()
plt.savefig("ql_ts.png")
plt.close()


print "here"
plt.figure()
plt.contourf(qt_mean, 200)
plt.colorbar()
plt.savefig("qt_ts.png")
plt.close()



plt.figure() 
plt.plot(t, np.ma.masked_where(cloud_top<-30000.0, cloud_top)) 
plt.savefig('cloud_top.png') 
plt.close() 


#print "here"
#plt.figure()
#plt.contourf(qr_mean,66)
#plt.savefig("qr_ts.png")
#plt.close()


print "here"
plt.figure() 
plt.plot(thetali_mean[:,0],z) 
plt.plot(np.mean(thetali_mean[:,-2:],axis=1),z)
#plt.ylim(0.0,20000.0) 
#plt.xlim(273.15,500.0)  
plt.savefig('thetali.png') 
plt.close() 


print "here"
plt.figure()
plt.plot(ql_mean[:,0],z)
plt.plot(np.mean(ql_mean[:,-2:],axis=1),z)
#plt.ylim(0.0,20000.0) 
#plt.xlim(273.15,500.0
plt.xlabel('qt [kg/kg]')
plt.ylabel('z [m]')  
plt.savefig('ql.png')
plt.close()



plt.figure()
#plt.plot(t0[:],z,'-k')
plt.plot(temperature_mean[:,0],z)
plt.plot(np.mean(temperature_mean[:,-2:],axis=1),z)
#plt.plot(t_gcm,z_gcm)
plt.ylim(0.0,25.6)
#plt.xlim(273.15,500.0)  
plt.grid() 
plt.xlabel('T [K]') 
plt.ylabel('Z [m]') 
plt.savefig('temperature.png')
plt.savefig('temperature.pdf')
plt.close()

plt.figure()
#plt.plot(t0[:],z,'-k')
plt.plot(s_mean[:,0],z)
plt.plot(np.mean(s_mean[:,-2:],axis=1),z)
#plt.ylim(0.0,20000.0)
#plt.xlim(273.15,500.0)
plt.grid()
plt.xlabel('T [K]')
plt.ylabel('Z [m]')
plt.savefig('s.png')
plt.close()


plt.figure()
plt.plot(dsdt_hadv_mean[:,0],z)
plt.plot(np.mean(dsdt_hadv_mean[:,-2:],axis=1),z)
#plt.ylim(0.0,20000.0) 
#plt.xlim(273.15,500.0)  
plt.grid()
plt.xlabel('ds_dt')
plt.ylabel('Z [m]')
plt.savefig('dsdt_hadv_mean.png')
plt.close()

plt.figure()
plt.plot(ls_subsidence[:,0],z)
plt.plot(np.mean(ls_subsidence[:,-2:],axis=1),z)
#plt.ylim(0.0,20000.0)
#plt.xlim(273.15,500.0)
plt.grid()
plt.xlabel('ls_subsidence')
plt.ylabel('Z [m]')
plt.savefig('ls_subsidence.png')
plt.close()

plt.figure()
plt.plot(grey_rad_heating_mean[:,0],z)
plt.plot(np.mean(grey_rad_heating_mean[:,-2:],axis=1),z)
#plt.ylim(0.0,20000.0) 
#plt.xlim(273.15,500.0)  
plt.grid()
plt.xlabel('ds_dt')
plt.ylabel('Z [m]')
plt.savefig('grey_rad_heating_mean.png')
plt.close()

plt.figure()
plt.plot(np.mean(ls_subs_dsdt[:,-t_min:],axis=1),z, label='subs')
#plt.plot(np.mean(ls_subs_dsdt[:,-360:-358],axis=1),z, label='subs')
#plt.plot(grey_rad_dsdt[:,0],z)
plt.plot(np.mean(grey_rad_dsdt[:,-t_min:],axis=1),z, label='radn')
#plt.plot(dsdt_hadv_mean[:,0],z)
plt.plot(np.mean(dsdt_hadv_mean[:,-t_min:],axis=1),z, label='hadv')
plt.plot(np.mean(dsdt_hadv_mean[:,-t_min:],axis=1) + np.mean(grey_rad_dsdt[:,-t_min:],axis=1) + np.mean(ls_subs_dsdt[:,-t_min:],axis=1), z, label='net')
plt.legend(loc='upper left') 

#plt.ylim(0.0,20000.0)
#plt.xlim(273.15,500.0)
plt.grid()
plt.xlabel('ds_dt')
plt.ylabel('Z [m]')
plt.savefig('ls_subs_dtdt_new.png')
plt.close()




#plt.figure()
#plt.contour(dsdt_hadv_mean[:,:] + grey_rad_dsdt[:,:] + ls_subs_dtdt[:,:],100)
#plt.savefig('heat_ts.png')
#plt.close()


print "here"
plt.figure()
plt.plot(cloud_fraction[:,0],z)
plt.plot(np.mean(cloud_fraction[:,-600:],axis=1),z)
#plt.ylim(0.0,15000.0)
#plt.xlim(2)
plt.savefig('cloud_fraction.png')
plt.close() 

plt.figure()
plt.plot(t, shf, label='Sensible')

tmin = np.min(t)
tmax = np.max(t)
plt.plot(t, lhf, label='Latent')
plt.hlines(lhf_gcm, tmin, tmax, 'g')
plt.hlines(shf_gcm, tmin, tmax, 'b')
plt.grid() 
plt.legend(loc='upper left') 
plt.savefig('surface_balance.png') 
plt.close() 

print "here"
plt.figure()
print "fig"
plt.plot(qt_mean[:,0],z)
plt.plot(np.mean(qt_mean[:,-2:],axis=1),z)
plt.xlabel('qt [kg/kg]') 
plt.ylabel('z [m]') 
plt.grid() 

print "plots"
#plt.ylim(0.0,12000.0)
#plt.xlim(273.15,375.0)
plt.savefig('qt.png')
print "save" 
rt_grp.close()
print "close" 
