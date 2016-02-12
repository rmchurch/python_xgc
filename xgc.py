
"""Module of the XGC loader for general use, taken from Loic's load_XGC_local for BES.

It reads the data from the simulation and remove the points not wanted.
This file is based on an other code written by Lei Shi (:download:`code <../../../../FPSDP/Plasma/XGC_Profile/load_XGC_profile.py>`).

"""

import numpy as np
import os.path
import glob
from scipy.interpolate import splrep, splev
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator

#convenience gateway to load XGC1 or XGCa data
def load(*args,**kwargs):
    file_path = os.path.join(args[0],'')

    if len(glob.glob(file_path+'xgc.3d*')) > 0:
        return xgc1Load(*args,**kwargs)
    
    elif len(glob.glob(file_path+'xgc.2d*')) > 0:
        return xgcaLoad(*args,**kwargs)

    else:
        raise ValueError('XGC files not found in '+file_path)


class _load(object):
    """Loader class for general use.

    The idea of this loader is to load all data, limiting spatially and temporally as user-specified.

    :param str xgc_path: Name of the directory containing the data
    :param int t_start: Time step at which starting the diagnostics
    :param int t_end: Time step at which stoping the diagnostics
    :param int dt: Interval between two time step that the diagnostics should compute
    :param np.array[...,2] limits: Mesh limits for the diagnostics. Give an np.array[2,2], 
        in this case, R/Z_min and R/Z_max are given (first index is for R,Z).\
    :param str kind: Order of the interpolation method (linear or cubic))
    """

    def __init__(self,xgc_path,t_start=1,t_end=None,dt=1,
        Rmin=None,Rmax=None,Zmin=None,Zmax=None,
        psinMin=None,psinMax=None,phi_start=0, phi_end=None,
        kind='linear'):
        """Copy all the input values and call all the functions that compute the equilibrium and the first
        time step.
        """

        print 'Loading XGC output data'
        
        self.xgc_path = os.path.join(xgc_path,'')  #get file_path, add path separator if not there
        self.mesh_file=self.xgc_path+'xgc.mesh'
        #check if files are in HDF5 or ADIOS format
        if os.path.exists(self.mesh_file+'.bp'):
            ext='.bp';
            import adios
            self.readCmd=lambda x,v: adiosreadvar(x+ext,v)
        elif os.path.exists(self.mesh_file+'.h5'):
            ext='.h5';
            import h5py
            self.readCmd=lambda x,v: h5py.File(x+ext,'r')[v][:]
        else:
            raise ValueError('No xgc.mesh file found')

        print 'from directory:'+ self.xgc_path

        #read in units file
        self.unit_file = self.xgc_path+'units.m'
        self.unit_dic = self.load_m(self.unit_file)

        #read in time
        self.oneddiag_file=self.xgc_path+'xgc.oneddiag'
        self.time = self.readCmd(self.oneddiag_file,'time')
        self.t_start=t_start
        self.t_end=t_end        
        if self.t_end is None: self.t_end=len(self.time)
        self.time = self.time[(self.t_start-1):(self.t_end):dt]
        self.time_steps = np.arange(self.t_start,self.t_end+1,dt) #1-based for file names
        self.tstep = self.unit_dic['sml_dt']*self.unit_dic['diag_1d_period']
        self.dt = self.tstep * dt
        self.Ntimes = len(self.time)


        # limits of the mesh in tokamak coordinates. Set to min,max of arrays in loadMesh()
        #if unspecified by user
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Zmin = Zmin
        self.Zmax = Zmax
        self.psinMin=psinMin
        self.psinMax=psinMax

        self.kind = kind
        
        
        #read in mesh, equilibrium data, and finally fluctuation data
        print 'Loading mesh and psi...'
        self.loadMesh()
        print 'mesh and psi loaded.'
        
        print 'Loading equilibrium...'
        self.loadEquil()
        print 'equlibrium loaded.'

    
    def load_m(self,fname):
        """load the whole .m file and return a dictionary contains all the entries.
        """
        f = open(fname,'r')
        result = {}
        for line in f:
            words = line.split('=')
            key = words[0].strip()
            value = words[1].strip(' ;\n')
            result[key]= float(value)
        f.close()
        return result

    def loadMesh(self):
        """load R-Z mesh and psi values, then create map between each psi 
           value and the series of points on that surface.
        """
        # get mesh R,Z and psin
        RZ = self.readCmd(self.mesh_file,'coordinates/values')
        R=RZ[:,0]
        Z=RZ[:,1]
        psi = self.readCmd(self.mesh_file,'psi')
        psin = psi/self.unit_dic['psi_x']
        tri=self.readCmd(self.mesh_file,'/cell_set[0]/node_connect_list') #already 0-based

        # set limits if not user specified
        if self.Rmin is None: self.Rmin=np.min(R)
        if self.Rmax is None: self.Rmax=np.max(R)
        if self.Zmin is None: self.Zmin=np.min(Z)
        if self.Zmax is None: self.Zmax=np.max(Z)
        if self.psinMin is None: self.psinMin=np.min(psin)
        if self.psinMax is None: self.psinMax=np.max(psin)

        #limit to the user-input ranges        
        self.rzInds = ( (R>=self.Rmin) & (R<=self.Rmax) & 
            (Z>=self.Zmin) & (Z<=self.Zmax) & 
            (psin>=self.psinMin) & (psin<=self.psinMax) )

        self.RZ = RZ[self.rzInds,:]
        self.psin = psin[self.rzInds]

        # psi interpolant
        fill_ = np.nan
        if self.kind == 'linear':
            self.psi_interp = LinearNDInterpolator(
                self.RZ, self.psin, fill_value=fill_)
        elif self.kind == 'cubic':
            self.psi_interp = CloughTocher2DInterpolator(
                self.RZ, self.psin, fill_value=fill_)
        else:
            raise NameError("The method '{}' is not defined".format(self.kind))

        #get the triangles which are all contained within the vertices defined by
        #the indexes igrid
        #find which triangles are in the defined spatial region
        tmp=self.rzInds[tri] #rzInds T/F array, same size as R
        goodTri=np.all(tmp,axis=1) #only use triangles who have all vertices in rzInds
        self.tri=tri[goodTri,:]
        #remap indices in triangulation
        indices=np.where(self.rzInds)[0]
        for i in range(len(indices)):
            self.tri[self.tri==indices[i]]=i


    def loadEquil(self):
        """Load equilibrium profiles and compute the interpolant
        """
        #read in 1D psin data
        self.psin1D = self.readCmd(self.oneddiag_file,'psi')
        if self.psin1D.ndim > 1: self.psin1D = self.psin1D[0,:]

        #read electron temperature
        try:
          etemp_par=self.readCmd(self.oneddiag_file,'e_parallel_mean_en_avg')
          etemp_per=self.readCmd(self.oneddiag_file,'e_perp_temperature_avg')
        except:
          etemp_par=self.readCmd(self.oneddiag_file,'e_parallel_mean_en_1d')
          etemp_per=selfreadCmd(self.oneddiag_file,'e_perp_temperature_1d')
        self.Te1D=(etemp_par[0,:]+etemp_per[0,:])*2./3

        #read electron density
        self.ne1D = self.readCmd(self.oneddiag_file,'e_gc_density_1d')[0,:]

        #create splines
        self.te0_sp = splrep(self.psin1D,self.Te1D,k=1)
        self.ne0_sp = splrep(self.psin1D,self.ne1D,k=1)


class xgc1Load(_load):
    def __init__(self,xgc_path,phi_start=0,phi_end=None,**kwargs):
        #call parent loading init, including mesh and equilibrium
        super(xgc1Load,self).__init__(xgc_path,**kwargs)

        #read in number of planes
        fluc_file0 = self.xgc_path + 'xgc.3d.' + str(self.time_steps[0]).zfill(5)
        self.Nplanes=self.readCmd(fluc_file0,'dpot').shape[1]
        self.phi_start=phi_start
        self.phi_end = phi_end
        if phi_end is None: self.phi_end=self.Nplanes
        self.Nplanes=self.phi_end-self.phi_start
        
        print 'Loading fluctuations...'
        self.loadFluc()
        print 'fluctuations loaded'

    def loadFluc(self):
        """Load non-adiabatic electron density and electrical static 
        potential fluctuations for 3D mesh.
        The required planes are calculated and stored in sorted array.
        fluctuation data on each plane is stored in the same order.
        Note that for full-F runs, the perturbed electron density 
        includes both turbulent fluctuations and equilibrium relaxation,
        this loading method doesn't differentiate them and will read all of them.
        
        """
        self.eden = np.zeros( (len(self.RZ[:,0]), self.Nplanes, self.Ntimes) )
        
        self.dpot = np.zeros( (len(self.RZ[:,0]), self.Nplanes, self.Ntimes) )
        
        for i in range(self.t_start,self.t_end+1):
            flucFile = self.xgc_path + 'xgc.3d.'+str(i).zfill(5)

            self.dpot[:,:,i-1] = self.readCmd(flucFile,'dpot')[self.rzInds,self.phi_start:(self.phi_end+1)]
            
            self.eden[:,:,i-1] = self.readCmd(flucFile,'eden')[self.rzInds,self.phi_start:(self.phi_end+1)]
        
        if self.Nplanes == 1:
            self.dpot = self.dpot.squeeze()
            self.eden = self.eden.squeeze()


    def calcNeTotal(self,psin=None):
        """Calculate the total electron at the wanted points.

        :param np.array[N] psin

        :returns: Total density
        :rtype: np.array[N]
        
        """
        
        if psin is None: psin=self.psin

        # temperature and density (equilibrium) on the psi mesh
        te0 = splev(psin,self.te0_sp)
        # avoid temperature <= 0
        te0[te0<np.min(self.Te1D)/10] = np.min(self.Te1D)/10
        ne0 = splev(psin,self.ne0_sp)
        ne0[ne0<np.min(self.ne1D)/10] = np.min(self.ne1D)/10
        

        #neAdiabatic = ne0*exp(dpot/te0)
        factAdiabatic = np.exp(np.einsum('i...,i...->i...',self.dpot,1./te0))
        neAdiabatic = np.einsum('i...,i...->i...',ne0,factAdiabatic)

        #ne = neAdiatbatice + dneKinetic
        ne = neAdiabatic + self.eden

        #TODO I've ignored checking whether dne<<ne0, etc. may want to add
        return ne




class xgcaLoad(_load):
    def __init__(self,*args,**kwargs):
        #call parent loading init, including mesh and equilibrium
        super().__init__(*args,**kwargs)

        print 'Loading f0 data...'
        self.loadf0()
        print 'f0 data loaded'

    def loadf0(self):
        ##f0 mesh data
        self.f0_file = self.xgc_path+'xgc.f0.mesh'
        #load velocity grid parallel velocity
        f0_nvp = self.readCmd(self.f0_file,'f0_nvp')
        self.nvpa = 2*f0_nvp+1 #actual # of Vparallel velocity pts (-vpamax,0,vpamax)
        self.vpamax = self.readCmd(self.f0_file,'f0_vp_max')
        self.vpar = np.linspace(-self.f0_vp_max,self.vpamax,self.nvpa)
        #load velocity grid perpendicular velocity
        f0_nmu = self.readCmd(self.f0_file,'f0_nmu')
        self.nvpe = f0_nmu + 1 #actual # of Vperp velocity pts (0,vpemax)
        self.vpemax = self.readCmd(self.f0_file,'f0_smu_max')
        self.vperp = np.linspace(0,self.f0_smu_max,self.nvpe)
        #load velocity grid density
        self.f0_ne = self.readCmd(self.f0_file,'f0_den')
        #load velocity grid electron and ion temperature
        f0_t_ev = self.readCmd(self.f0_file,'f0_t_ev')
        self.f0_Te = f0_t_ev[0,:]
        self.f0_Ti = f0_t_ev[1,:]

        self.f0_grid_vol_vonly = self.readCmd(self.f0_file,'f0_grid_vol_vonly')


