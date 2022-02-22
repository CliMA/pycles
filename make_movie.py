import pickle as pkl
import numpy as np
import pylab as plt
from os import listdir


path = 'Output.1_00x_lon_144.1_00x_lon_144/Visualization/'
fs = listdir(path)[-200:]

print fs

count = 0


var = 'w'
v_min = 999999999.0
v_max = -999999999.0
for f in fs:
    fpath = path + f
    fh = open(fpath, 'rb')
    d = pkl.load(fh)
    data = d[var]
    v_min =  np.min([v_min, np.amin(data)])
    v_max = np.max([v_max, np.amax(data)])

    fh.close()

    print v_min, v_max

vextreme = np.max([np.abs(v_min), np.abs(v_max)])
v_min = -vextreme
v_max = vextreme

for f in fs:
    print float(count)/len(fs)


    fpath = path + f
    fh = open(fpath, 'rb')
    d = pkl.load(fh)

    z = d['zp_half'][3:-3]
    ql = d['ql']
    qi = d['qi']
    qs = d['qs']
    qc = ql + qi + qs
    qc = qc - np.mean(qc, axis=0)[np.newaxis,:] 
    x=np.arange(d[var].shape[0]) * 1000.0

    print np.shape(ql), np.shape(qi), np.shape(qs)
    ax = plt.figure(figsize=( 64.0/4, 16.0/4.))
    ax.patch.set_facecolor('black') 
    
    plt.contourf(x, z,  d[var].T, 100, vmin=v_min, vmax=v_max , cmap=plt.cm.seismic)
    plt.contour(x, z, qc.T, levels=[0.00001], colors='b')
    plt.clim(v_min, v_max)
    plt.ylim(0.0,15000.0)
    plt.savefig('./movie/' + str(10000000 + count) + '.png')
    plt.close()

    fh.close()



    count += 1
