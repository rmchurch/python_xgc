
"""Module of the XGC loader for general use, taken from Loic's load_XGC_local for BES.

It reads the data from the simulation and remove the points not wanted.
This file is based on an other code written by Lei Shi (:download:`code <../../../../FPSDP/Plasma/XGC_Profile/load_XGC_profile.py>`).

"""

import numpy as np
import os.path
import sys
import glob
from scipy.interpolate import splrep, splev
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from matplotlib.tri import Triangulation
import adios2 as ad
import h5py
import matplotlib.pyplot as plt
from scipy.io import matlab
from scipy.io.matlab import loadmat
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
from scipy import stats


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

    INPUTS
    :param str xgc_path: Name of the directory containing the data

    OPTIONAL
    :param int t_start: Time step at which starting the diagnostics [1-based indexed (to match file)]
    :param int t_end: Time step at which stoping the diagnostics [1-based indexed (to match file)]
    :param int dt: Interval between two time step that the diagnostics should compute
    :param float Rmin: Limit minimum major radius of data
    :param float Rmax: Limit maximum major radius of data
    :param float Zmin: Limit minimum vertical coordinate of data
    :param float Zmax: Limit maximum vertical coordinate of data
    :param float psinMin: Limit minimum normalized poloidal flux ("psi-normal") of data
    :param float psinMax: Limit maximum normalized poloidal flux ("psi-normal") of data
    :param int phi_start: Toroidal plane index to start data [0-based indexed]
    :param int phi_end: Toroidal plane index to stop data [0-based indexed]


    :param str kind: Order of the interpolation method (linear or cubic))
    """
    
    def __init__(self,xgc_path,t_start=1,t_end=None,dt=1,
        Rmin=None,Rmax=None,Zmin=None,Zmax=None,
        psinMin=None,psinMax=None,thetaMin=None,thetaMax=None, 
        phi_start=0, phi_end=None,
        kind='linear',
        skiponeddiag=False):
        """Copy all the input values and call all the functions that compute the equilibrium and the first
        time step.
        """
        def openAdios(x):
            return ad.open(str(x)+'.bp','r')
        def readAdios(x,v,inds=Ellipsis):
            if '/' in v: v = '/'+v
            #v = '/'+v #this may be necessary for older xgc files
            if type(x) is ad.File:
                assert(v in x.available_variables().keys())
                return x.read(v)[inds]
            else:
                f = openAdios(x)
                assert(v in f.available_variables().keys())
                data = f.read(v)[inds]
                f.close()
                return data
        
        def openAdios2(x):
            return ad.open(str(x)+'.bp','r')
        def readAdios2(x,v,inds=Ellipsis):
            if '/' in v: v = '/'+v
            #v = '/'+v #this may be necessary for older xgc files
            if type(x) is ad.File:
                if x.current_step() >= x.steps():
                    x.close()
                    x = openAdios2(x)
                f = x
            else:
                f = openAdios2(x)
        
            nstep = int(f.available_variables()[v]['AvailableStepsCount'])
            nsize = f.available_variables()[v]['Shape']
            if nstep==1:
                data = f.read(v)[inds]
            elif nsize != '': #mostly xgc.oneddiag
                nsize = int(nsize)
                data = f.read(v,start=[0], count=[nsize], step_start=0, step_count=nstep)
            else: #mostly xgc.oneddiag
                data = f.read(v,start=[], count=[], step_start=0, step_count=nstep)
            
            if not type(x) is ad.File:
                f.close()
            
            return data

        
        def openHDF5(x):
            return h5py.File(str(x)+'.h5','r')
        def readHDF5(x,v,inds=Ellipsis):
            if type(x) is h5py.File:
                return x[v][...][inds]       
            else:
                f = openHDF5(x)
                data = f[v][inds]
                f.close()
                return data
        
        print('Loading XGC output data')
        
        self.xgc_path = os.path.join(xgc_path,'')  #get file_path, add path separator if not there
        self.mesh_file=self.xgc_path+'xgc.mesh'
        #check if files are in HDF5 or ADIOS format
        if os.path.exists(self.mesh_file+'.bp'):
            import adios2 as ad
            self.openCmd=openAdios2
            self.readCmd=readAdios2
        elif os.path.exists(self.mesh_file+'.h5'):
            import h5py
            self.openCmd=openHDF5
            self.readCmd=readHDF5
        else:
            raise ValueError('No xgc.mesh file found')
        
        print('from directory:'+ self.xgc_path)
        #read in units file
        self.unit_file = self.xgc_path+'units.m'
        self.unit_dic = self.load_m(self.unit_file)
        
        self.inputused_file = self.xgc_path+'fort.input.used'
        self.ptl_mass,self.ptl_charge = self.load_mass_charge(self.inputused_file)
        
        #read in time
        self.oneddiag_file=self.xgc_path+'xgc.oneddiag'
        self.mask1d = self.oned_mask()
        try:
            self.time = np.array(self.readCmd(self.oneddiag_file,'time')[self.mask1d],ndmin=1)
        except:
            self.time = [0]
        if t_start is None: t_start=1
        assert t_start > 0, "t_start must be greater than 0 (1-based index)"
        self.t_start=int(t_start)
        if t_end is None: t_end=len(self.time)
        self.t_end=int(t_end)
        dt = int(dt)
        self.time = self.time[(self.t_start-1):(self.t_end):dt]
        self.time_steps = np.arange(self.t_start,self.t_end+1,dt) #1-based for file names
        self.tstep = self.unit_dic['sml_dt']*self.unit_dic['diag_1d_period']
        self.Ntimes = len(self.time)
        
        #magnetics file
        self.bfield_file=self.xgc_path+'xgc.bfield'
        print('Loading magnetics...')
        self.loadBfield()
        print('\tmagnetics loaded.')

        
        # limits of the mesh in tokamak coordinates. Set to min,max of arrays in loadMesh()
        #if unspecified by user
        self.Rmin = self.unit_dic['eq_x_r'] if 'x' in str(Rmin).lower() else Rmin
        self.Rmax = self.unit_dic['eq_x_r'] if 'x' in str(Rmax).lower() else Rmax
        self.Zmin = self.unit_dic['eq_x_z'] if 'x' in str(Zmin).lower() else Zmin
        self.Zmax = self.unit_dic['eq_x_z'] if 'x' in str(Zmax).lower() else Zmax
        self.psinMin=psinMin
        self.psinMax=psinMax
        
        self.thetaMin=thetaMin
        self.thetaMax=thetaMax
        
        self.kind = kind
        
          
        #read in mesh, equilibrium data, and finally fluctuation data
        print('Loading mesh and psi...')
        self.loadMesh()
        self.load_fluxavg()
        print('\tmesh and psi loaded.')
        
        #TODO: This isnt right yet, need to instead find saddlepoint
        #could do using gradient in LinearTriinterpolator or Cubic 
        #some units.m dont have eq_x_r,eq_x_z, approximate
        # if not ('eq_x_z' in self.unit_dic):
        #     import matplotlib._tri as _tri
        #     from matplotlib.tri import Triangulation
        #     triObj = Triangulation(self.RZ[:,0],self.RZ[:,1],self.tri)
        #     C = _tri.TriContourGenerator(triObj.get_cpp_triangulation(),self.psin)
        #     RZsep = C.create_contour(1.0)[0]
        #     xind = np.argmin(RZsep[:,1])
        #     self.unit_dic['eq_x_r'] = RZsep[xind,0]
        #     self.unit_dic['eq_x_z'] = RZsep[xind,1]
        
        
        if not skiponeddiag:
            print('Loading equilibrium...')
            self.load_oneddiag()
            print('\tequlibrium loaded.')
        else:
            print('Skipping equilibrium...')
    
    
    
    
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
    
    
    def load_mass_charge(self,fname):
        """load particle masses from fort.input.used (currently ptl_e_mass_au not in units.m
        """
        #TODO: Not general, someimtes comma on end of number
        global ptl_mass
        proton_mass = 1.6720e-27
        e_charge = 1.6022e-19
        ptl_mass = np.array([5.446e-4,2.0])*proton_mass
        ptl_charge = np.array([-1.0,1.0])*e_charge
        try:
            f = open(fname,'r')
            result = {}
            for line in f:
                if 'PTL_E_MASS_AU' in line:
                    ptl_mass[0] = float(line.split(',')[0].split()[-1]) * proton_mass
                if 'PTL_MASS_AU' in line:
                    ptl_mass[1] = float(line.split(',')[0].split()[-1]) * proton_mass
                if 'PTL_E_CHARGE_AU' in line:
                    ptl_charge[0] = float(line.split(',')[0].split()[-1]) * e_charge
                if 'PTL_CHARGE_AU' in line:
                    ptl_charge[1] = float(line.split(',')[0].split()[-1]) * e_charge
        except:
            pass            
        return ptl_mass,ptl_charge#will return default values
    
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
        tri=self.readCmd(self.mesh_file,'cell_set[0]/node_connect_list') #already 0-based
        node_vol=self.readCmd(self.mesh_file,'node_vol')
        
        #get LFS X-point, where Bpol is null
        if 'eq_x_r' not in self.unit_dic.keys():
            Bpol = np.sqrt(np.sum(self.bfield[:,0:2]**2.,axis=1))
            ind = np.argmin(Bpol[10:])+10
            self.unit_dic['eq_x_r'],self.unit_dic['eq_x_z'] = RZ[ind,:]
        
        theta = 180./np.pi*np.arctan2(RZ[:,1]-self.unit_dic['eq_axis_z'],RZ[:,0]-self.unit_dic['eq_axis_r'])
        self.theta_x = 180./np.pi*np.arctan2(self.unit_dic['eq_x_z']-self.unit_dic['eq_axis_z'],self.unit_dic['eq_x_r']-self.unit_dic['eq_axis_r'])
        try:
            wall_nodes = self.readCmd(self.mesh_file,'wall_nodes')-1 #-1 for 0-based index. Usually for XGCa
        except:
            try:
                wall_nodes = self.readCmd(self.mesh_file,'grid_wall_nodes')-1 #-1 for 0-based index. This is sometimes for XGC1
            except:
                print('no wall_nodes')
                wall_nodes = np.array([-1])
        
        # set limits if not user specified
        if self.Rmin is None: self.Rmin=np.min(R)
        if self.Rmax is None: self.Rmax=np.max(R)
        if self.Zmin is None: self.Zmin=np.min(Z)
        if self.Zmax is None: self.Zmax=np.max(Z)
        if self.psinMin is None: self.psinMin=np.min(psin)
        if self.psinMax is None: self.psinMax=np.max(psin)
        if self.thetaMin is None: self.thetaMin=np.min(theta)
        if self.thetaMax is None: self.thetaMax=np.max(theta)
        
        #limit to the user-input ranges        
        self.rzInds = ( (R>=self.Rmin) & (R<=self.Rmax) & 
            (Z>=self.Zmin) & (Z<=self.Zmax) & 
            (psin>=self.psinMin) & (psin<=self.psinMax) &
            (theta>=self.thetaMin) & (theta<=self.thetaMax) )
        
        self.RZ = RZ[self.rzInds,:]
        self.psin = psin[self.rzInds]
        self.node_vol = node_vol[self.rzInds]
        self.theta = theta[self.rzInds]
        self.wall_nodes = np.where(np.in1d(np.where(self.rzInds)[0],wall_nodes))[0]
        self.bfield[self.rzInds,:]
        
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
        if np.sum(self.rzInds)<R.size:
            tmp=self.rzInds[tri] #rzInds T/F array, same size as R
            goodTri=np.all(tmp,axis=1) #only use triangles who have all vertices in rzInds
            self.tri=tri[goodTri,:]
            #remap indices in triangulation
            indices=np.where(self.rzInds)[0]
            for i in range(len(indices)):
                self.tri[self.tri==indices[i]]=i
        else:
            self.tri = tri
        
        self.triObj = Triangulation(self.RZ[:,0],self.RZ[:,1],self.tri)
        
        self.flux_surfaces()
        
        
    def load_oneddiag(self):
        """Load all oneddiag quantities. Rename required equilibrium profiles and compute the interpolant
        """
        
        #read in all data from xgc.oneddiag
        f1d = self.openCmd(self.oneddiag_file)
        oneddiag={}
        #class structtype(): pass
        try:
            keys = f1d.var.keys()
        except:
            #keys = [key for key in f1d.keys()]
            #adios2
            keys = [key for key in f1d.available_variables().keys()]
        keys.sort()
        for key in keys:
            data = self.readCmd(f1d,key)
            if data.ndim==2: data = data[self.mask1d,:]
            oneddiag[key]=data
        self.oneddiag = oneddiag
        
        #read in volume information (used for radial fluxes calculation)
        #try:
        fv = self.openCmd('xgc.volumes')
        self.vol1d = self.readCmd(fv,'diag_1d_vol')
        psi_mks = self.oneddiag['psi_mks']
        if len(psi_mks.shape)>1:
            psi_mks = psi_mks[-1,...]
        self.dvol_dpsi = self.vol1d/np.gradient(psi_mks)
        #except:
        #    print("\t\tdV/dpsi calculation failed")

        #TODO: Decide if should remove this legacy renaming
        #modify 1d psin data
        self.psin1d = self.oneddiag['psi']
        if self.psin1d.ndim > 1: self.psin1d = self.psin1d[0,:]
        
        #read n=0,m=0 potential
        try:
            self.psin001d = self.oneddiag['psi00_1d']/self.unit_dic['psi_x']
        except:
            self.psin001d = self.oneddiag['psi00']/self.unit_dic['psi_x']
        if self.psin001d.ndim > 1: self.psin001d = self.psin001d[0,:]
        self.pot001d = self.oneddiag['pot00_1d']
        
        #read electron temperature
        try:
            itemp_par=self.oneddiag['i_parallel_mean_en_avg']
            itemp_per=self.oneddiag['i_perp_temperature_avg']
        except:
            itemp_par=self.oneddiag['i_parallel_mean_en_df_1d']
            itemp_per=self.oneddiag['i_perp_temperature_df_1d']
        self.Ti1d=(itemp_par+itemp_per)*2./3
        
        try:
            etemp_par=self.oneddiag['e_parallel_mean_en_avg']
            etemp_per=self.oneddiag['e_perp_temperature_avg']
            self.Te1d=(etemp_par+etemp_per)*2./3
            #read electron density
            self.ne1d = self.oneddiag['e_gc_density_df_1d']
        except:
            try:
                etemp_par=self.oneddiag['e_parallel_mean_en_df_1d']
                etemp_per=self.oneddiag['e_perp_temperature_df_1d']
                self.Te1d=(etemp_par+etemp_per)*2./3
                #read electron density
                self.ne1d = self.oneddiag['e_gc_density_df_1d']
            except: #ion only sim
                etemp_par = itemp_par
                etemp_per = itemp_per
                self.Te1d=(etemp_par+etemp_per)*2./3
                #read electron density
                self.ne1d = np.apply_along_axis(lambda a: np.interp(self.psin1d,self.psin001d,a),1,self.pot001d)/self.Te1d
        
        #create splines for t=0 data
        self.ti0_sp = splrep(self.psin1d,self.Ti1d[0,:],k=1)
        self.te0_sp = splrep(self.psin1d,self.Te1d[0,:],k=1)
        self.ne0_sp = splrep(self.psin1d,self.ne1d[0,:],k=1)
        
    
    def loadBfield(self):
        """Load magnetic field
        """
        try:
            self.bfield = self.readCmd(self.bfield_file,'node_data[0]/values')[...]
        except:
            try:
                self.bfield = self.readCmd(self.bfield_file,'/node_data[0]/values')[...]
            except:
                self.bfield = self.readCmd(self.bfield_file,'bfield')[...]
        
        
    def oned_mask(self):
        """Match oned data to 3d files, in cases of restart.
           Use this on oneddiag variables, e.g. n_e1d = ad.file('xgc.oneddiag.bp','e_gc_density_1d')[mask1d,:]
        """
        try:
            step = self.readCmd(self.oneddiag_file,'step')
            dstep = step[1] - step[0]
            
            idx = np.arange(step[0]/dstep,step[-1]/dstep+1)
            
            mask1d = np.zeros(idx.shape,dtype=np.int32)
            for (i,idxi) in enumerate(idx):
                mask1d[i] = np.where(step == idxi*dstep)[0][-1] #get last occurence
        except:
            mask1d = Ellipsis #pass variables unaffected
        
        return mask1d
    
    
    def flux_surfaces(self): 
        psin_surf = []
        nextnodes = []
        nextnode = 0
        try:
            self.psin_surf = self.readCmd(self.mesh_file,'psi_surf')/self.unit_dic['psi_x']
        except:
            for i in range(10000):
                psin_surf += [self.psin[nextnode]]
                nextnodes += [nextnode]
                inds = np.where(self.triObj.edges==nextnode)
                #inds is (Npts,2) array, and nextnode could appear in either.
                #what we want is not where nextnode appears, but where its neighbor is
                #so flip inds[1] (these are all 0 or 1, so you want the other index for the neighbor)
                possiblenodes = self.triObj.edges[inds[0],~inds[1]]
                nextnode = possiblenodes[np.argmax(self.RZ[possiblenodes,0])]
                if (np.any(self.wall_nodes==nextnode)) | (self.psin[nextnode]<psin_surf[-1]):
                    break
            self.psin_surf = np.array(psin_surf)
    
    
    def calc_bary(self,R,Z):
        """Given points, calculate their barycentric coordinates"""
        p = np.zeros(R.shape+(3,))
        trii = np.zeros(R.shape+(3,),dtype=int)-1
        
        #find triangles corresponding to R,Z position. Equivalent to tr_save in XGC1
        if ~hasattr(self,'trifinder'):
            self.trifinder = self.triObj.get_trifinder()
        indtri = self.trifinder(R,Z)
        #exclude out of bounds triangles (indtri==-1)
        ininds = np.where(indtri>-1)[0]
        #get triangles vertices these points are enclosed by
        trii[ininds,:] = self.triObj.triangles[indtri[ininds],:]
        
        #next, grab the vertices of these triangles, and put into an array T=[[x1,x2,x3],[y1,y2,y3],[1,1,1]],
        #and stack the regular grid into xi=[R,Z,1]
        x = self.triObj.x[trii[ininds,:]]
        y = self.triObj.y[trii[ininds,:]]
        T = np.concatenate( (x[...,np.newaxis],y[...,np.newaxis],np.ones(x.shape)[...,np.newaxis]),axis=2).swapaxes(1,2)
        xi = np.concatenate( (R[ininds,np.newaxis],Z[ininds,np.newaxis],np.ones(R[ininds].shape)[...,np.newaxis]),axis=1)
        #finally, solve system T bary = xi
        p[ininds,:] = np.linalg.solve(T,xi)
        return p,trii
    
    def load_fluxavg(self):
        try:
            self.fluxavg_file = self.xgc_path+'xgc.fluxavg'
            nelement = self.readCmd(self.fluxavg_file,'nelement')
            eindex = self.readCmd(self.fluxavg_file,'eindex')-1
            norm1d = self.readCmd(self.fluxavg_file,'norm1d')
            value = self.readCmd(self.fluxavg_file,'value')
            npsi = self.readCmd(self.fluxavg_file,'npsi')
            self.fluxavg_mat = self.create_sparse_xgc(nelement, eindex, value, m=nelement.size, n=npsi).T
            self.fluxAvg = self.fluxAvgNew
        except:
            self.fluxAvg = self.fluxAvgOld

    def fluxAvgNew(self,data,psin_inds=None):
        dataAvg = self.fluxavg_mat.dot(data)
        return dataAvg

    def fluxAvgOld(self,data,psin_inds=None):
        if psin_inds is None: psin_inds = np.arange(self.psin_surf.size,dtype=int) 
        dataAvg = np.zeros((psin_inds.size,))
        for (i,p) in enumerate(self.psin_surf[psin_inds]):
            pinds = np.where(np.abs(self.psin-p)<1e-4)[0]
            if p<1:pinds = pinds[self.RZ[pinds,1]>self.unit_dic['eq_x_z']]
            dataAvg[i] = self.flux_section_avg(data,pinds)
        return dataAvg
    
    def flux_section_avg(self,data,pinds):
        return np.sum(data[pinds,:]*self.node_vol[pinds,np.newaxis])/np.sum(self.node_vol[pinds]*data.shape[1])
    
    
    def loadf0mesh(self):
        ##f0 mesh data
        self.f0mesh_file = self.xgc_path+'xgc.f0.mesh'
        #load velocity grid parallel velocity
        f0_nvp = self.readCmd(self.f0mesh_file,'f0_nvp')
        self.nvpa = 2*f0_nvp+1 #actual # of Vparallel velocity pts (-vpamax,0,vpamax)
        self.vpamax = self.readCmd(self.f0mesh_file,'f0_vp_max')
        #load velocity grid perpendicular velocity
        f0_nmu = self.readCmd(self.f0mesh_file,'f0_nmu')
        self.nvpe = f0_nmu + 1 #actual # of Vperp velocity pts (0,vpemax)
        self.vpemax = self.readCmd(self.f0mesh_file,'f0_smu_max')
        self.vpa, self.vpe, self.vpe1 = self.create_vpa_vpe_grid(f0_nvp,f0_nmu,self.vpamax,self.vpemax)
        #load velocity grid density
        self.f0_ne = self.readCmd(self.f0mesh_file,'f0_den')
        #load velocity grid electron and ion temperature
        self.f0_T_ev = self.readCmd(self.f0mesh_file,'f0_T_ev')
        self.f0_Te = self.f0_T_ev[0,:]
        self.f0_Ti = self.f0_T_ev[1,:]
        
        self.f0_grid_vol_vonly = self.readCmd(self.f0mesh_file,'f0_grid_vol_vonly')
        
        
    def create_vpa_vpe_grid(self,f0_nvp, f0_nmu, f0_vp_max, f0_smu_max):
        """Create velocity grid vectors"""
        vpe=np.linspace(0,f0_smu_max,f0_nmu+1) #dindgen(nvpe+1)/(nvpe)*vpemax
        vpe1=vpe.copy()
        vpe1[0]=vpe[1]/3.
        vpa=np.linspace(-f0_vp_max,f0_vp_max,2*f0_nvp+1)
        return (vpa, vpe, vpe1)

    def create_sparse_xgc(self,nelement,eindex,value,m=None,n=None):
        """Create Python sparse matrix from XGC data"""
        from scipy.sparse import csr_matrix
        
        if m is None: m = nelement.size
        if n is None: n = nelement.size
        
        #format for Python sparse matrix
        indptr = np.insert(np.cumsum(nelement),0,0)
        indices = np.empty((indptr[-1],))
        data = np.empty((indptr[-1],))
        for i in range(nelement.size):
                indices[indptr[i]:indptr[i+1]] = eindex[i,0:nelement[i]]
                data[indptr[i]:indptr[i+1]] = value[i,0:nelement[i]]
        #create sparse matrices
        spmat = csr_matrix((data,indices,indptr),shape=(m,n))
        return spmat
    
            
    ######## ANALYSIS ###################################################################
    def calcMoments(self,ind=1):
        """Calculate moments from the f0 data
        """
    
        self.f0_file = self.xgc_path + 'xgc.f0.'+str(ind).zfill(5)
        #read distribution data
        e_f  = self.readCmd(self.f0_file,'e_f')[:,self.rzInds,:]
        i_f  = self.readCmd(self.f0_file,'i_f')[:,self.rzInds,:]
        
        self.ne2d,self.Vepar2d,self.Te2d,self.Tepar2d,self.Teperp2d = self.calcMoments1(e_f,0)
        self.ni2d,self.Vipar2d,self.Ti2d,self.Tipar2d,self.Tiperp2d = self.calcMoments1(i_f,1)
        #TODO: Add calculation for fluxes, Vpol (requires more info)
        
        return (self.ne2d,self.Vepar2d,self.Te2d,self.Tepar2d,self.Teperp2d,\
                self.ni2d,self.Vipar2d,self.Ti2d,self.Tipar2d,self.Tiperp2d)
    
    
    
    def calcMoments1(self,f0,isp):
        """Calculate moments from the f0 data
        """
        mass,charge,vspace_vol,volfac,vth = self.moments_params(isp)
        
        #calculate moments of f0 using einsum for fast(er) calculation
        den2d = np.einsum('...ijk,ik,j->...j',f0,volfac,vspace_vol)
        Vpar2d = vth*np.einsum('k,...ijk,ik,j->...j',self.vpa,f0,volfac,vspace_vol)/den2d
        
        prefac = mass/(2.*np.abs(charge))
        Tpar2d = 2.*prefac*( vth**2.*np.einsum('k,...ijk,ik,j->...j',self.vpa**2.,f0,volfac,vspace_vol)/den2d - Vpar2d**2. )
        Tperp2d = prefac*vth**2.*np.einsum('i,...ijk,ik,j->...j',self.vpe**2.,f0,volfac,vspace_vol)/den2d
        T2d = (Tpar2d + 2.*Tperp2d)/3.
        
        return (den2d,Vpar2d,T2d,Tpar2d,Tperp2d)
    
    def moments_params(self,isp):
        """Return mass,charge,velocity space volume,discrete cell correction, and thermal velocity
        """
        
        # Extract species of interest (0 electrons, 1 ions)
        mass = self.ptl_mass[isp]
        charge = self.ptl_charge[isp]
        
        vspace_vol = self.f0_grid_vol_vonly[isp,:]
        
        #discrete cell correction
        volfac = np.ones((self.vpe.size,self.vpa.size))
        volfac[0,:]=0.5 #0.5 for where ivpe==0
        
        Tev = self.f0_T_ev[isp,:]
        vth=np.sqrt(np.abs(charge)*Tev/mass)
        
        return mass,charge,vspace_vol,volfac,vth
    
    
    def create_f0para(self,f0,isp):
        """Create parallel distribution function
        """
        #discrete cell correction
        volfac = np.ones((self.vpe.size,self.vpa.size))
        volfac[0,:]=0.5 #0.5 for where ivpe==0
        
        # Extract species of interest (0 electrons, 1 ions)
        mass = self.ptl_mass[isp]
        charge = self.ptl_charge[isp]
        
        vspace_vol = self.f0_grid_vol_vonly[isp,:]
        return np.einsum('ijk,ik->jk',f0,volfac)*vspace_vol[:,np.newaxis]

    def plot_profiles1d(self,timeinds=[0,-1]):
        """Plot the basic plasma profiles"""
        fig,ax = plt.subplots(1,3,sharex=True,figsize=(10,4))
        for ti in timeinds:
            if ti<0: ti = self.ne1d.shape[0]+ti
            ax[0].plot(self.psin1d,self.ne1d[ti,:],label="t=%d"%(ti+1))
            ax[1].plot(self.psin1d,self.Te1d[ti,:],label="t=%d"%(ti+1))
            ax[2].plot(self.psin1d,self.Ti1d[ti,:],label="t=%d"%(ti+1))
        for i in range(3):
            ax[i].set_xlabel('$\psi_N$')
            ax[i].legend()
        ax[0].set_title('$n_e$')
        ax[1].set_title('$T_e$')
        ax[2].set_title('$T_i$')
        
        
    
class xgc1Load(_load):
    def __init__(self,xgc_path,phi_start=0,phi_end=None,skip_fluc=False,**kwargs):
        #call parent loading init, including mesh and equilibrium
        #super().__init__(*args,**kwargs)
        super(xgc1Load,self).__init__(xgc_path,**kwargs)
        
        #read in number of planes
        fluc_file0 = self.xgc_path + 'xgc.3d.' + str(self.time_steps[0]).zfill(5)
        self.Nplanes=self.readCmd(fluc_file0,'dpot').shape[1]
        # assert isinstance(phi_start,int), "phi_start must be a plane index (Int)"
        # assert isinstance(phi_end,int), "phi_end must be a plane index (Int)"
        self.phi_start=int(phi_start)
        if phi_end is None: phi_end=self.Nplanes-1
        self.phi_end = int(phi_end)
        self.Nplanes=self.phi_end-self.phi_start+1
        
        if not skip_fluc:
            print('Loading fluctuations...')
            self.loadFluc()
            print('fluctuations loaded')
        
        if not skip_fluc:
            print('Loading flux data...')
            #self.loadf3d()
            print('flux surfaces loaded')
    
    
    def loadFluc(self):
        """Load non-adiabatic electron density, electrical static 
        potential fluctuations, and n=0 potential for 3D mesh.
        The required planes are calculated and stored in sorted array.
        fluctuation data on each plane is stored in the same order.
        Note that for full-F runs, the perturbed electron density 
        includes both turbulent fluctuations and equilibrium relaxation,
        this loading method doesn't differentiate them and will read all of them.
        
        """
        #from read_fluc_single import read_fluc_single #gives no module error
        
        self.eden = np.zeros( (len(self.RZ[:,0]), self.Nplanes, self.Ntimes) )
        self.dpot = np.zeros( (len(self.RZ[:,0]), self.Nplanes, self.Ntimes) )
        self.pot0 = np.zeros( (len(self.RZ[:,0]), self.Ntimes) )
        
        #def read_fluc_single(i,readCmd,xgc_path,rzInds,phi_start,phi_end): #seems to be a different version of the below method
         #   import adios
          #  flucFile = adios.file(xgc_path + 'xgc.3d.'+str(i).zfill(5)+'.bp')
           # dpot1 = flucFile['dpot'][rzInds,phi_start:(phi_end+1)]
            #pot01 = flucFile['pot0'][rzInds]
            #eden1 = flucFile['eden'][rzInds,phi_start:(phi_end+1)]
            #return i,dpot1,pot01,eden1
           
             
        def read_fluc_single(i,readCmd,xgc_path,rzInds,phi_start,phi_end):
            import adios2 as ad
            flucFile = ad.open(xgc_path + 'xgc.3d.'+str(i).zfill(5)+'.bp','r')
            dpot1 = readCmd(flucFile,'dpot',inds=(rzInds,)+(slice(phi_start,phi_end+1),) )#[self.rzInds,self.phi_start:(self.phi_end+1)]
            pot01 = readCmd(flucFile,'pot0',inds=(rzInds,) )#[rzInds]
            eden1 = readCmd(flucFile,'eden',inds=(rzInds,)+(slice(phi_start,phi_end+1),) )#[self.rzInds,self.phi_start:(self.phi_end+1)]
            return i,dpot1,pot01,eden1
        
        #import time
        #start = time.time() #ipyparallel
        #def read_fluc_single(i,openCmd,xgc_path,rzInds,phi_start,phi_end): #for ipyparallel
        #    print 'went in method'
        #    flucFile = openCmd(xgc_path + 'xgc.3d.'+str(i).zfill(5))
            #flucFile = h5py.File(xgc_path + 'xgc.3d.'+str(i).zfill(5)+'.h5')
        #    start = time.time()
        #    dpot1 = flucFile['dpot'][rzInds,phi_start:(phi_end+1)]
        #    pot01 = flucFile['pot0'][rzInds]
        #    eden1 = flucFile['eden'][rzInds,phi_start:(phi_end+1)]
        #    flucFile.close()
        #    print 'Read time: '+str(time.time()-start)
        #    return i,dpot1,pot01,eden1
        
        
        #try:
        #import ipyparallel as ipp
        #print "went into ipyparallel"
        #rc = ipp.Client()
        
        #dview = rc[:] #load balanced view cant be used because I need to push data
        #dview.use_dill() #before was getting pickle error for Ellipsis, not sure where the Ellipsis is
        #with dview.sync_imports():
        #    import adios
        #    import h5py
        #    import time
        #    import sys
        #    sys.path.append(os.environ['HOME']+'/python_xgc/')
            #from read_fluc_single import read_fluc_single 
        #dview.push(dict(xgc_path=self.xgc_path,rzInds=self.rzInds,phi_start=self.phi_start,phi_end=self.phi_end))
                    #from read_fluc_single import read_fluc_single 
        #out = dview.map_sync(lambda i: read_fluc_single(i,self.openCmd,self.xgc_path,self.rzInds,self.phi_start,self.phi_end),range(self.t_start,self.t_end+1))
                    
        #for i in range(self.Ntimes): #self.t_start,self.t_end+1
        #    _,self.dpot[:,:,i],self.pot0[:,i],self.eden[:,:,i] = out[i]
        #print 'Read time: '+str(time.time()-start)
        
        #except:
        for i in range(self.Ntimes): 
            sys.stdout.write('\r\tLoading file ['+str(i)+'/'+str(self.Ntimes)+']')
            _,self.dpot[:,:,i],self.pot0[:,i],self.eden[:,:,i] = read_fluc_single(self.t_start + i,self.readCmd,self.xgc_path,self.rzInds,self.phi_start,self.phi_end)
                
        #for i in range(self.Ntimes): #same as the for loop above
        #    sys.stdout.write('\r\tLoading file ['+str(i)+'/'+str(self.Ntimes)+']')
        #    f = self.openCmd(self.xgc_path+'xgc.3d.'+str(i+1).zfill(5))
        #    self.dpot[:,:,i] = self.readCmd(f,'dpot')[self.rzInds,self.phi_start:(self.phi_end+1)]
        #    self.eden[:,:,i] = self.readCmd(f,'eden')[self.rzInds,self.phi_start:(self.phi_end+1)]
        #    self.pot0[:,i] = self.readCmd(f,'pot0')[self.rzInds]
        #    f.close()
        
        if self.Nplanes == 1:
            self.dpot = self.dpot.squeeze()
            self.eden = self.eden.squeeze()
    
    def loadf3d(self):
        #from read_fluc_single import read_fluc_single #gives no module error
        
        self.i_T_perp = np.zeros( (len(self.RZ[:,0]), self.Nplanes,self.Ntimes) )
        self.i_E_para = np.zeros( (len(self.RZ[:,0]), self.Nplanes,self.Ntimes) )
        self.i_u_para = np.zeros( (len(self.RZ[:,0]), self.Nplanes,self.Ntimes) )
        self.i_den = np.zeros( (len(self.RZ[:,0]), self.Nplanes,self.Ntimes) )
        self.e_T_perp = np.zeros( (len(self.RZ[:,0]), self.Nplanes,self.Ntimes) )
        self.e_E_para = np.zeros( (len(self.RZ[:,0]), self.Nplanes,self.Ntimes) )
        self.e_u_para = np.zeros( (len(self.RZ[:,0]), self.Nplanes,self.Ntimes) )
        self.e_den = np.zeros( (len(self.RZ[:,0]), self.Nplanes,self.Ntimes) )
        
        #def read_fluc_single(i,readCmd,xgc_path,rzInds,phi_start,phi_end): #seems to be a different version of the below method
         #   import adios
          #  flucFile = adios.file(xgc_path + 'xgc.3d.'+str(i).zfill(5)+'.bp')
           # dpot1 = flucFile['dpot'][rzInds,phi_start:(phi_end+1)]
            #pot01 = flucFile['pot0'][rzInds]
            #eden1 = flucFile['eden'][rzInds,phi_start:(phi_end+1)]
            #return i,dpot1,pot01,eden1
        # import time #ipypparallel
        # start = time.time()
        # def read_fluc_single(i,openCmd,xgc_path,rzInds,phi_start,phi_end):
        #     flucFile = openCmd(xgc_path + 'xgc.f3d.'+str(i).zfill(5))
        #     #flucFile = h5py.File(xgc_path + 'xgc.3d.'+str(i).zfill(5)+'.h5')
            
        #     i_T_perp = flucFile['i_T_perp'][rzInds,phi_start:(phi_end+1)]
        #     i_E_para = flucFile['i_E_para'][rzInds,phi_start:(phi_end+1)]
        #     i_u_para = flucFile['i_u_para'][rzInds,phi_start:(phi_end+1)]
        #     i_den = flucFile['i_den'][rzInds,phi_start:(phi_end+1)]
        #     e_T_perp = flucFile['e_T_perp'][rzInds,phi_start:(phi_end+1)]
        #     e_E_para = flucFile['e_E_para'][rzInds,phi_start:(phi_end+1)]
        #     e_u_para = flucFile['e_u_para'][rzInds,phi_start:(phi_end+1)]
        #     e_den = flucFile['e_den'][rzInds,phi_start:(phi_end+1)]
        #     flucFile.close()
        #     print 'Read time: '+str(time.time()-start)
        #     return i_T_perp,i_E_para,i_u_para,i_den,e_T_perp,e_E_para,e_u_para,e_den
           
        # import ipyparallel as ipp
        # print "went into ipyparallel"
        # rc = ipp.Client()
        
        # dview = rc[:] #load balanced view cant be used because I need to push data
        # dview.use_dill() #before was getting pickle error for Ellipsis, not sure where the Ellipsis is
        # with dview.sync_imports():
        #     import adios
        #     import h5py
        #     import time
        #     import sys
        #     sys.path.append(os.environ['HOME']+'/python_xgc/')
        #     #from read_fluc_single import read_fluc_single 
        # dview.push(dict(xgc_path=self.xgc_path,rzInds=self.rzInds,phi_start=self.phi_start,phi_end=self.phi_end))
        #             #from read_fluc_single import read_fluc_single 
        # out = dview.map_sync(lambda i: read_fluc_single(i,self.openCmd,self.xgc_path,self.rzInds,self.phi_start,self.phi_end),range(self.t_start,self.t_end+1))
                    
        # for i in range(self.Ntimes): #self.t_start,self.t_end+
        #     self.i_T_perp[:,:,i],self.i_E_para[:,:,i],self.i_u_para[:,:,i],self.i_den[:,:,i],self.e_T_perp[:,:,i],self.e_E_para[:,:,i],self.e_u_para[:,:,i],self.e_den[:,:,i] = out[i]
        # print 'Read time: '+str(time.time()-start)
        
        def read_fluc_single(i,readCmd,xgc_path,rzInds,phi_start,phi_end):
           import adios2 as ad
           f3dFile = ad.open(xgc_path + 'xgc.f3d.'+str(i).zfill(5)+'.bp','r')
           i_T_perp = readCmd(f3dFile,'i_T_perp',inds=(rzInds,)+(slice(phi_start,phi_end+1),) )#[self.rzInds,self.phi_start:(self.phi_end+1)]
           i_E_para = readCmd(f3dFile,'i_E_para',inds=(rzInds,)+(slice(phi_start,phi_end+1),)  )#[rzInds]
           i_u_para = readCmd(f3dFile,'i_u_para',inds=(rzInds,)+(slice(phi_start,phi_end+1),)  )#[rzInds]            
           i_den = readCmd(f3dFile,'i_den',inds=(rzInds,)+(slice(phi_start,phi_end+1),) )#[self.rzInds,self.phi_start:(self.phi_end+1)]
           e_T_perp = readCmd(f3dFile,'e_T_perp',inds=(rzInds,)+(slice(phi_start,phi_end+1),) )#[self.rzInds,self.phi_start:(self.phi_end+1)]
           e_E_para = readCmd(f3dFile,'e_E_para',inds=(rzInds,)+(slice(phi_start,phi_end+1),) )#[self.rzInds,self.phi_start:(self.phi_end+1)]
           e_u_para = readCmd(f3dFile,'e_u_para',inds=(rzInds,)+(slice(phi_start,phi_end+1),) )#[self.rzInds,self.phi_start:(self.phi_end+1)]
           e_den = readCmd(f3dFile,'e_den',inds=(rzInds,)+(slice(phi_start,phi_end+1),) )#[self.rzInds,self.phi_start:(self.phi_end+1)]
           return i_T_perp,i_E_para,i_u_para,i_den,e_T_perp,e_E_para,e_u_para,e_den
        
        i=0 #read in the last time step file since f3d is not over time
        sys.stdout.write('\r\tLoading file ['+str(i)+'/'+str(self.Ntimes)+']')
        self.i_T_perp[:,:,0],self.i_E_para[:,:,i],self.i_u_para[:,:,0],self.i_den[:,:,0],self.e_T_perp[:,:,0],self.e_E_para[:,:,0],self.e_u_para[:,:,0],self.e_den[:,:,0] = read_fluc_single(self.t_start+i,self.readCmd,self.xgc_path,self.rzInds,self.phi_start,self.phi_end)
                
        
        #calculate full temperature
        self.i_T = (self.i_T_perp+(self.i_E_para- (ptl_mass[1]/2*self.i_u_para**2)/(1.609e-19 )))*2./3
        self.e_T = (self.e_T_perp+(self.e_E_para- (ptl_mass[0]/2*self.e_u_para**2)/(1.609e-19 )))*2./3
        
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
        te0[te0<np.min(self.Te1d)/10] = np.min(self.Te1d)/10
        ne0 = splev(psin,self.ne0_sp)
        ne0[ne0<np.min(self.ne1d)/10] = np.min(self.ne1d)/10
        
        
        #neAdiabatic = ne0*exp(dpot/te0)
        factAdiabatic = np.exp(np.einsum('i...,i...->i...',self.dpot,1./te0))
        self.neAdiabatic = np.einsum('i...,i...->i...',ne0,factAdiabatic)
        
        #ne = neAdiatbatic + dneKinetic
        self.n_e = self.neAdiabatic + self.eden
        
        #TODO I've ignored checking whether dne<<ne0, etc. may want to add
        return self.n_e
    
    def calcPotential(self):
        self.pot = self.pot0[:,np.newaxis,:] + self.dpot
        return self.pot
    
    
    def hist2dline1(self,x,y,bins,range=None,cmap='Reds',minmax=False):
        x = x.flatten()
        y = y.flatten()
        goodInds = np.where( ~np.isnan(x) & ~np.isnan(y) )[0]
        (cnts,xedges,yedges) = np.histogram2d(x[goodInds],y[goodInds],bins=bins,range=range)
        return cnts, xedges,yedges
    
    def hist2dline2(self,x,y,bins,range=None,cmap='Reds',minmax=False):
        x = x.flatten()
        y = y.flatten()
        goodInds = np.where( ~np.isnan(x) & ~np.isnan(y) )[0]
        (cnts,xedges,yedges) = np.histogram2d(x[goodInds],y[goodInds],bins=bins,range=range)
        xedgeMid = (xedges[1::] + xedges[:-1])/2.
        yedgeMid = (yedges[1::] + yedges[:-1])/2.
        yAvg = np.sum(yedgeMid[np.newaxis,:]*cnts,axis=1)/np.sum(cnts,axis=1)
        dMin=[]
        dMax=[]
        if minmax:
            dMin = np.empty((xedges.size-1,))
            dMax = np.empty((xedges.size-1,))
            for i in np.arange(xedges.size-1):
                if (y[(x>=xedges[i]) & (x<xedges[i+1])].size==0):
                    print(i)
                    dMin[i]=np.nan
                    dMax[i]=np.nan
                else:
                    dMin[i] = y[(x>=xedges[i]) & (x<xedges[i+1])].min()
                    dMax[i] = y[(x>=xedges[i]) & (x<xedges[i+1])].max()
            for i in np.arange(xedges.size-1):
                if np.isnan(dMin[i])==True:
                    dMin[i]=(dMin[i-1]+dMin[i+1])/2
                    dMax[i]=(dMax[i-1]+dMax[i+1])/2
            plt.plot(xedgeMid,dMin,'k--',xedgeMid,dMax,'k--')
        return xedgeMid,yAvg,dMin,dMax
    def kfSpectrum(self,L,time,frames,noNormalize=False, noFilter=False, window=None):
        
        if window is not None:
            #default to Hanning, add others later
            wL=0.5*(1-np.cos(2.*np.pi*np.arange(L.size)/L.size))
            wTime=0.5*(1-np.cos(2.*np.pi*np.arange(time.size)/time.size))
            win2 = wL[:,np.newaxis]*wTime[np.newaxis,:] #or should this be matrix mult?
            frames = frames*win2
        
        NFFT = 2**np.ceil(np.log2(frames.shape[0:2])).astype(int)
        # %%%Create k and f arrays
        # %create frequency array
        Fs=1./np.mean(np.diff(time));
        f=Fs/2*np.linspace(0,1,NFFT[1]/2+1) # %highest frequency 0.5 sampling rate (Nyquist)
        
        # %METHOD 1: No anti-aliasing
        # %create k array
        kmax=np.pi/np.min(np.diff(L))
        k=kmax*np.linspace(-1,1,NFFT[0])
        kfspec=np.fft.fftshift(np.fft.ifft(np.fft.fft(frames,n=NFFT[1],axis=1),n=NFFT[0],axis=0))  #%fftshift since Matlab puts positive frequencies first
        kfspec=kfspec[:,NFFT[1]/2-1:,...]
        
        # % %METHOD 2: Anti-aliasing
        # % kfspec=fftshift(fft(frames,NFFT(2),2));
        # % %for now, remove negative frequency components
        # % kfspec=kfspec(:,NFFT(2)/2:end);
        # % kmin=pi/(L(end)-L(1));
        # % kmax=pi/min(diff(L))*0.85;
        # % k=[linspace(-kmax,-kmin,NFFT(1)/2-1) 0 linspace(kmin,kmax,NFFT(1)/2)];
        # % kfspec=exp(i*k(:)*L(:)')*kfspec;
        
        # %%%normalize, S(k|w)=S(k,w)/S(w)
        if not noNormalize:
            kfspec=np.abs(kfspec)/np.sum(np.abs(kfspec),axis=0)[np.newaxis,:]
        
        # %%%OPTIONAL: filtering (smooths images)
        if not noFilter:
            kfspec = gaussian_filter(np.abs(kfspec), sigma=5)
        
        return k,f,kfspec
    
    
class xgcaLoad(_load):
    def __init__(self,xgc_path,**kwargs):
        #call parent loading init, including mesh and equilibrium
        #super().__init__(*args,**kwargs)
        super(xgcaLoad,self).__init__(xgc_path,**kwargs)
        
        print('Loading f0 data...')
        self.loadf0mesh()
        print('f0 data loaded')
    
    def load2D():
        self.iden = np.zeros( (len(self.RZ[:,0]), self.Ntimes) )
        
        self.dpot = np.zeros( (len(self.RZ[:,0]), self.Ntimes) )
        self.pot0 = np.zeros( (len(self.RZ[:,0]), self.Ntimes) )
        self.epsi = np.zeros( (len(self.RZ[:,0]), self.Ntimes) )
        self.etheta = np.zeros( (len(self.RZ[:,0]), self.Ntimes) )
        
        
        for i in range(self.Ntimes):
            flucFile = self.xgc_path + 'xgc.2d.'+str(t_start+i).zfill(5)
            
            self.iden[:,i] = self.readCmd(flucFile,'iden',inds=(self.rzInds,))#[self.rzInds]
            
            self.dpot[:,i] = self.readCmd(flucFile,'dpot',inds=(self.rzInds,))#[self.rzInds]
            self.pot0[:,i] = self.readCmd(flucFile,'pot0',inds=(self.rzInds,))#[self.rzInds]
            self.epsi[:,i] = self.readCmd(flucFile,'epsi',inds=(self.rzInds,))#[self.rzInds]
            self.etheta[:,i] = self.readCmd(flucFile,'etheta',inds=(self.rzInds,))#[self.rzInds]
            
            
            
            
class gengridLoad():
    def __init__(self,file_path):
        self.file_path = file_path
        
        #read in the grid data from node file
        #TODO Read in ele and poly components also
        f = open(self.file_path,'r')
        Nlines = int(f.readline().split()[0])
        self.RZ = np.empty([Nlines,2])
        for i,line in enumerate(f):
            if i >= Nlines: break
            self.RZ[i,0:2] = np.array(line.split()[1:3],dtype='float')
