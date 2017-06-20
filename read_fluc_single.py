#!/usr/bin python

import adios
import h5py
import time

def read_fluc_single(i,xgc_path,rzInds,phi_start,phi_end):
    flucFile = adios.file(xgc_path + 'xgc.3d.'+str(i).zfill(5)+'.bp')
    #flucFile = h5py.File(xgc_path + 'xgc.3d.'+str(i).zfill(5)+'.h5')
    start = time.time()
    dpot1 = flucFile['dpot'][rzInds,phi_start:(phi_end+1)]
    pot01 = flucFile['pot0'][rzInds]
    eden1 = flucFile['eden'][rzInds,phi_start:(phi_end+1)]
    flucFile.close()
    print 'Read time: '+str(time.time()-start)
    return i,dpot1,pot01,eden1

