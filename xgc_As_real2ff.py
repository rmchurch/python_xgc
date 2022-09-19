# coding: utf-8
import numpy as np
import xgc
from matplotlib.tri import Triangulation, LinearTriInterpolator
import matplotlib.pyplot as plt

#limit data to the [Rmin,Rmax,Zmin,Zmax] box, and read only the first two toroidal planes
#Rmin=2.2
#Rmax=2.31
#Zmin=-0.25
#Zmax=0.4
phi_start=0
phi_end=31
t_start = 1
t_end = 1

fileDir='/p/xgc/rhager/summit/dave_pugmire/analysis'

#load XGC data, and calculate normalized electron density
#loader=xgc.load(fileDir,Rmin=Rmin,Rmax=Rmax,Zmin=Zmin,Zmax=Zmax,phi_start=phi_start,phi_end=phi_end)
loader=xgc.load(fileDir,phi_start=phi_start,phi_end=phi_end,t_start=t_start,t_end=t_end,dt=1,skiponeddiag=True)


#setup mesh grid
Ri=np.linspace (loader.Rmin,loader.Rmax,400)
Zi=np.linspace (loader.Zmin,loader.Zmax,400)
RI,ZI=np.meshgrid(Ri,Zi)
#interpolate using the TriInterpolator class. Should be what tricontourf() uses
triObj=Triangulation(loader.RZ[:,0],loader.RZ[:,1],loader.tri)
sepInds = np.where(np.abs(loader.psin-1.0)<1e-4)[0]


# Calculate grad(As) and transform As and grad(As) to
# field-following representation
dAs        = loader.GradAll(loader.As[:,:,0])
As_phi_ff  = loader.conv_real2ff(loader.As[:,:,0])
dAs_phi_ff = -loader.conv_real2ff(dAs)

# Write Adios file with perturbed vector potential in
# field-following representation
import adios2 as ad
fbp   = ad.open("test.bp","w")
nphi  = dAs_phi_ff.shape[0]
nnode = dAs_phi_ff.shape[1]
fbp.write("nphi",np.array([nphi]))
fbp.write("nnode",np.array([nnode]))
# For some reason the numpy data layout for these variables is not
# C-style --> make contiguous.
dum = np.ascontiguousarray(As_phi_ff[:,:,:,0])
fbp.write("As_phi_ff",dum, dum.shape, [0]*len(dum.shape), dum.shape)
dum = np.ascontiguousarray(dAs_phi_ff)
fbp.write("dAs_phi_ff",dum, dum.shape, [0]*len(dum.shape), dum.shape)
fbp.close()


plt.figure(1)
tci=LinearTriInterpolator(triObj,As_phi_ff[0,:,0,0])
out=tci(RI,ZI)
fac=0.25
colra=np.arange(np.min(out)*fac,np.max(out)*fac,fac*np.abs(np.max(out)-np.min(out))*0.01)
plt.contourf(RI,ZI,out,levels=colra)
plt.colorbar()
plt.xlabel('R [m]')
plt.ylabel('Z [m]')
plt.show()


