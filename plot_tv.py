import pylab as plt
import numpy as np
import netCDF4 as nc
import cPickle as pkl
t_min =1 

#f = './Stats.GCMVarying.nc'
f = './Output.GCMVarying.9ddb9/stats/Stats.GCMVarying.nc'
f_pkl = './forcing/1_00x/f_data_tv_90.pkl'
f_pkl = file(f_pkl, 'r')
d_pkl = pkl.load(f_pkl)
f_pkl.close()


print d_pkl.keys()

u_gcm  = d_pkl['u'][:,:]
v_gcm  = d_pkl['v'][:,:]

qflux_gcm  = d_pkl['qflux'][:]
lwdn_gcm  = d_pkl['lwdn_sfc'][:]
lwup_gcm  = d_pkl['lwup_sfc'][:]
swdn_gcm  = d_pkl['swdn_sfc'][:]
swdn_toa_gcm =  d_pkl['swdn_toa'][:]
rad_heating_gcm = d_pkl['temp_rad'][:,:]
fino_heating_gcm = d_pkl['temp_fino'][:,:]
vadv_heating_gcm = d_pkl['temp_vadv'][:,:]
total_heating_gcm = d_pkl['temp_total'][:,:]
hadv_heating_gcm = d_pkl['temp_hadv'][:,:]
param_heating_gcm = d_pkl['temp_param'][:,:]
diffusion_heating_gcm = d_pkl['temp_diffusion'][:,:]
t_gcm = d_pkl['temp'][:,:]
shum_gcm = d_pkl['shum'][:,:]
z_gcm = np.mean(d_pkl['zfull'][:],axis=0) 
shf_gcm = d_pkl['shf_flux'][:]
lhf_gcm = d_pkl['lhf_flux'][:]
sst_gcm = d_pkl['ts'][:]


time_gcm = np.arange(shf_gcm.shape[0], dtype=np.double) * 0.25




vadv_shum_gcm = d_pkl['dt_qg_vadv'][:,:]
hadv_shum_gcm = d_pkl['dt_qg_hadv'][:,:]

print z_gcm.shape
print hadv_shum_gcm.shape

rt_grp = nc.Dataset(f,'r') 


profiles = rt_grp['profiles'] 
reference = rt_grp['reference'] 
timeseries = rt_grp['timeseries'] 
 
z = reference['zp_half'][:] 
sst = timeseries['surface_temperature'][:] 
t = timeseries['t'][:]/(3600.0*24.0)






time_gcm = time_gcm[time_gcm <= np.max(t)]
n_time_gcm = time_gcm.shape[0]
n_time_gcm_start = n_time_gcm/2



cloud_top = timeseries['cloud_top'][:] 

shf = timeseries['shf_surface_mean'][:]
lhf = timeseries['lhf_surface_mean'][:]
lwdn_sfc = timeseries['srf_lw_flux_down'][:]
swdn_sfc = timeseries['srf_sw_flux_down'][:]
swup_sfc = timeseries['srf_sw_flux_up'][:]
lwup_sfc = timeseries['srf_lw_flux_up'][:]

u_mean = profiles['u_mean'][:,:].T
v_mean = profiles['v_mean'][:,:].T
ql_mean = profiles['ql_mean'][:,:].T

try:
    qr_mean = profiles['qr_mean'][:, :].T
except:
    qr_mean = None
try:
    qr_mean = profiles['qrain_mean'][:,:].T
except:
    qr_mean = None


qv_mean = profiles['qv_mean'][:,:].T
qt_mean = profiles['qt_mean'][:,:].T
s_mean = profiles['s_mean'][:,:].T
cloud_fraction = profiles['cloud_fraction'][:,:].T
thetali_mean = profiles['thetali_mean'][:,:].T
temperature_mean = profiles['temperature_mean'][:,:].T
dsdt_hadv_mean = profiles['ls_dsdt_hadv'][:,:].T
sw_flux_down = profiles['sw_flux_down'][:,:].T
lw_flux_down = profiles['lw_flux_down'][:,:].T
lw_flux_up = profiles['lw_flux_up'][:,:].T
ls_subsidence = profiles['ls_subsidence'][:,:].T
t0 = reference['temperature0'][:]
p0 = reference['p0'][:]
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

#precip_dqtdt = profiles['dqtdt_precip_mean'][:,:].T


plt.close()


n_les_start = qt_mean.shape[1]/2

plt.figure(figsize=(12,6))
plt.plot(t,  -shf, '-r', label='LES SHF')
plt.plot(t,  -lhf, '-b', label='LES LHF')
plt.plot(t, lwdn_sfc - lwup_sfc, '-g', label='LW NET')
plt.plot(t, swdn_sfc - swup_sfc, '-y', label='SW')
plt.plot(t, -shf -lhf + (lwdn_sfc - lwup_sfc) + (swdn_sfc - swup_sfc) - np.mean(qflux_gcm),  '-*k', label='Net Flux')


plt.plot(time_gcm, -shf_gcm[:n_time_gcm], '-or', label='GCM SHF')
plt.plot(time_gcm, -lhf_gcm[:n_time_gcm], '-ob', label='GCM LHF')
plt.plot(time_gcm, swdn_gcm[:n_time_gcm], '-oy', label='GCM SW Net')
plt.plot(time_gcm, lwdn_gcm[:n_time_gcm] - lwup_gcm[:n_time_gcm], '-og', label='GCM LW Net')

plt.plot(time_gcm,  -shf_gcm[:n_time_gcm] -lhf_gcm[:n_time_gcm] + swdn_gcm[:n_time_gcm] + (lwdn_gcm[:n_time_gcm] - lwup_gcm[:n_time_gcm])  - np.mean(qflux_gcm), '-ok')



plt.grid()
plt.legend(loc='upper left')
plt.savefig('sfc_fluxes.png')
print "LES Net: ", np.mean(-shf -lhf + (lwdn_sfc - lwup_sfc) + (swdn_sfc - swup_sfc) - np.mean(qflux_gcm)), "GCM Net: ", np.mean(-shf_gcm[:n_time_gcm] -lhf_gcm[:n_time_gcm] + swdn_gcm[:n_time_gcm] + (lwdn_gcm[:n_time_gcm] - lwup_gcm[:n_time_gcm])  - np.mean(qflux_gcm))
print "LES SW: ", np.mean(swdn_sfc - swup_sfc), "GCM SW: ", np.mean(swdn_gcm[:n_time_gcm])
print "LES LW: ", np.mean(lwdn_sfc - lwup_sfc), "GCM LW: ", np.mean(lwdn_gcm[:n_time_gcm] - lwup_gcm[:n_time_gcm])
print "LES SHF: ", -np.mean(shf), "GCM SHF: ", -np.mean(shf_gcm[:n_time_gcm])
print "LES LHF: ", -np.mean(lhf), "GCM LHF: ", -np.mean(lhf_gcm[:n_time_gcm])
print "LES Bowen: ", np.mean(shf/lhf), "GCM Bowen: ", np.mean(shf_gcm[:n_time_gcm]/lhf_gcm[:n_time_gcm])


plt.close()
plt.figure()
plt.subplot(211)
plt.boxplot([lhf, lhf_gcm])
plt.grid()
plt.subplot(212)
plt.boxplot([shf, shf_gcm])
plt.grid()
plt.savefig('box_plots.pdf')
plt.close()
#plt.show()



plt.figure()
plt.plot(t,  shf/lhf, '-r', label='B LES')
plt.plot(time_gcm, shf_gcm[:n_time_gcm]/lhf_gcm[:n_time_gcm], '--r', label='B GCM')
plt.grid()
plt.legend(loc='upper left')
plt.savefig('bowen.png')




plt.figure()
plt.plot(t, sst[:], '-r', label='LES')
plt.plot(time_gcm, sst_gcm[:n_time_gcm], '--r', label = 'GCM')
plt.legend(loc='upper left')
plt.grid()
plt.savefig("sst.png")

plt.figure()

t_mesh, z_mesh  = np.meshgrid(t, z )
t_mesh, p_mesh = np.meshgrid(t, p0)

z /= 1000.0
z_gcm /=1000.0
balance_time = -1
plt.plot(np.mean(vadv_shum_gcm[:,:] + hadv_shum_gcm[:,:],axis=0) , z_gcm, 'or',label='GCM NET', linewidth=4.0)
#plt.plot(ls_dqtdt_eddy[:,balance_time] + ls_subs_dqtdt[:,balance_time] + ls_dqtdt_hadv[:,balance_time], z , '-k', label='LES Net', linewidth=3.0, alpha=0.75)
#plt.plot(hadv_shum_gcm , z_gcm, '+r', label='hadv')
#plt.plot(vadv_shum_gcm[0,:] , z_gcm, '*r', label='vadv')
#plt.plot(ls_dqtdt_hadv[:,balance_time],z, '--k', label='LES hadv', linewidth=3.0)
#plt.plot(ls_subs_dqtdt[:,balance_time],z, '-.k', label='LES subs', linewidth=3.0)
plt.plot(np.mean(ls_subs_dqtdt[:,:] + ls_dqtdt_hadv[:,:],axis=1),z, '-.k', label='LES Net', linewidth=3.0)
#plt.plot(ls_dqtdt_eddy[:,balance_time],z, ':k', label='LES eddy', linewidth=3.0)

plt.grid()
plt.ylim(0.0,25.6)
plt.xlabel('Specific Humidity Tendency kg/(kg s)')
plt.ylabel('height')
#plt.plot(ls_subs_dqtdt[:,0], '-k', label='subs')

plt.legend(loc='upper left')
plt.savefig('balance_terms_qt.png')
plt.savefig('balance_terms_qt.pdf')
plt.close()



plt.figure()
plt.contourf(t, z, u_mean[:,:], 100)
plt.colorbar()
plt.grid()
plt.savefig('u.png')
plt.close()


plt.figure()
plt.contourf(t, z, v_mean[:,:], 100)
plt.colorbar()
plt.grid()
plt.savefig('v.png')
plt.close()

plt.figure()
plt.contourf(t, z, np.sqrt(v_mean[:,:] * v_mean[:,:] + u_mean[:,:] * u_mean[:,:]), 100)
plt.colorbar()
plt.grid()
plt.savefig('speed.png')
plt.close()

plt.figure()
plt.plot(t,np.sqrt(v_mean[0,:] * v_mean[0,:] + u_mean[0,:] * u_mean[0,:]))
plt.grid()
plt.savefig('speed_ts.png')
plt.close()



plt.figure()
plt.plot(t0, z, linewidth=2.5, label='T0 LES')
plt.plot(temperature_mean[:,0], z , linewidth=2.5, label='T LES')
plt.plot(np.mean(t_gcm, axis=0) ,z_gcm, 'or', label='T GCM')
plt.legend()
plt.ylabel('Height [m]')
plt.xlabel('Temperature [K]')
plt.savefig('t_comp.pdf')
#plt.show()
plt.close()


balance_time = 1
plt.figure()
plt.plot(vadv_heating_gcm[0,:] + hadv_heating_gcm[0,:]  + fino_heating_gcm[0,:]  + rad_heating_gcm[0,:] , z_gcm, 'or', label='GCM Net')
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

plt.close() 
plt.figure() 
plt.plot(vadv_heating_gcm[0,:], z_gcm) 
plt.plot(ls_subs_dtdt[:,balance_time] , z, '-.k',linewidth=3.0, label=' LES vadv')
plt.savefig('subs_terms.png') 
plt.close()






#plt.figure()
#plt.contour(precip_dqtdt,levels=[-0.0000001, 0.0])
#plt.colorbar()
#plt.savefig('precip_rate.png')
#plt.close()



print "here"
plt.figure(figsize=(12,6))
plt.contourf(t_mesh, z_mesh/1000.0, ql_mean, 200)
plt.colorbar()
plt.grid()
plt.savefig("ql_ts.png")
plt.close()


#print "here"
#plt.figure(figsize=(12,6))
#plt.contourf(t_mesh, z_mesh/1000.0, qr_mean, 200)
#plt.colorbar()
#plt.grid()
#plt.savefig("qr_ts.png")
#plt.close()



print "here"
plt.figure(figsize=(12,6))
plt.contourf(t_mesh, z_mesh/1000.0, cloud_fraction, 200)
plt.colorbar()
plt.grid()
plt.savefig("cc_ts.png")
plt.close()


print "here"
plt.figure()
#plt.plot(ql_mean[:,0],z)
plt.plot(np.mean(cloud_fraction[:,n_les_start:],axis=1),z)
#plt.ylim(0.0,20000.0)
#plt.xlim(273.15,500.0
plt.xlabel('qt [kg/kg]')
plt.ylabel('z [m]')
plt.savefig('cc.png')
plt.close()


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
#plt.plot(ql_mean[:,0],z)
plt.plot(np.mean(ql_mean[:,n_les_start:],axis=1),z)
#plt.ylim(0.0,20000.0) 
#plt.xlim(273.15,500.0
plt.xlabel('qt [kg/kg]')
plt.ylabel('z [m]')  
plt.savefig('ql.png')
plt.close()



plt.figure()
#plt.plot(t0[:],z,'-k')
plt.plot(np.mean(t_gcm[:n_time_gcm,:],axis=0),z_gcm, label='GCM Mean')
plt.plot(np.mean(temperature_mean[:,:],axis=1),z, label='LES Mean')
plt.plot(temperature_mean[:,-1],z, label='LES Final')

#plt.plot(t_gcm,z_gcm)
plt.ylim(0.0,25.6)
#plt.xlim(273.15,500.0)  
plt.grid()
plt.legend()
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






plt.figure(figsize=(12,6))

vmax = np.amax(ls_subsidence)
vmin = np.amin(ls_subsidence)
vbnd = np.max([np.abs(vmax), np.abs(vmin)])

plt.contourf(t, z, ls_subsidence[:,:],100, vmin=-vbnd, vmax=vbnd, cmap=plt.cm.coolwarm)
plt.clim(-vbnd, vbnd)
#plt.ylim(0.0,20000.0)
#plt.xlim(273.15,500.0)
plt.grid()
plt.title('Subsidence [m/s]')
plt.xlabel('Time [days]')
plt.ylabel('Z [km]')
plt.colorbar()
plt.savefig('ls_subsidence.png')
plt.close()


print "here"
plt.figure()
print "fig"
print n_time_gcm
plt.plot(np.mean(shum_gcm[n_time_gcm_start:n_time_gcm,:],axis=0),z_gcm, '-k', label='GCM Mean')
plt.plot(np.mean(shum_gcm[:,:],axis=0),z_gcm, '--k', label='GCM Full')
plt.plot(np.mean(qt_mean[:,n_les_start:],axis=1),z, '-b', label="LES Mean")
plt.plot(qt_mean[:,-1], z, '--b', label='LES Final')
plt.ylim(0.0,25.6)
plt.xlabel('qt [kg/kg]')
plt.ylabel('z [m]')
plt.grid()
plt.legend()
plt.savefig('qt_mean.pdf')
plt.close()



print "here"
plt.figure(figsize=(12,6))
plt.contourf(t_mesh, p_mesh, qt_mean, 200)
plt.colorbar()
plt.gca().invert_yaxis()
plt.savefig("qt_ts.png")
plt.close()


import sys; sys.exit()

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



print "plots"
#plt.ylim(0.0,12000.0)
#plt.xlim(273.15,375.0)
plt.savefig('qt.png')
print "save" 
rt_grp.close()
print "close" 
