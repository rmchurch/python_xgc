# coding: utf-8
import numpy as np
import xgc
from matplotlib.tri import Triangulation, LinearTriInterpolator
import matplotlib.pyplot as plt

#limit data to the [Rmin,Rmax,Zmin,Zmax] box, and read only the first two toroidal planes
Rmin=2.2
Rmax=2.31
Zmin=-0.25
Zmax=0.4
phi_start=0
phi_end=1

fileDir='/global/project/projectdirs/m499/jlang/particle_pinch/'

#load XGC data, and calculate normalized electron density
loader=xgc.load(fileDir,Rmin=Rmin,Rmax=Rmax,Zmin=Zmin,Zmax=Zmax,phi_start=phi_start,phi_end=phi_end)

#plot the poloidal mesh
plt.figure(1)
plt.triplot(loader.RZ[:,0],loader.RZ[:,1],loader.tri)


#calculate the totoal electron density, and normalize to first time frame
#n_e will be size [Nverts,Nplanes,Ntimes], where Nverts is the number of 
#poloidal plane unstructured mesh vertices, Nplanes the number of toroidal planes, and Ntimes the number of time slices
n_e=loader.calcNeTotal()
neNorm=np.einsum('i...j,i...->i...j',n_e,1./n_e[:,:,0])

#setup mesh grid
Ri=np.linspace (loader.Rmin,loader.Rmax,400)
Zi=np.linspace (loader.Zmin,loader.Zmax,400)
RI,ZI=np.meshgrid(Ri,Zi)

#interpolate using the TriInterpolator class. Should be what tricontourf() uses
triObj=Triangulation(loader.RZ[:,0],loader.RZ[:,1],loader.tri)
tci=LinearTriInterpolator(triObj,neNorm[:,0,80]) #interpolate for the index 0 toroidal plane, index 80 time slice
out=tci(RI,ZI)

sepInds = np.where(np.abs(loader.psin-1.0)<1e-4)[0]

plt.figure(2)
plt.contourf(RI,ZI,out,100)
plt.clim([0,3])
plt.plot(loader.RZ[sepInds,0],loader.RZ[sepInds,1],'w--')
plt.colorbar()
plt.xlabel('R [m]')
plt.ylabel('Z [m]')
plt.show()
