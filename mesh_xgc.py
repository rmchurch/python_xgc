# coding: utf-8
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt; plt.ion()
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
from scipy import interpolate
import os

def mtanh(x,c0,c1,c2,c3,c4):
    z=2.*(c0-x)/c1
    res=0.5*(c2-c3)*( (1+c4*z)*np.exp(z)-np.exp(-z) )/( np.exp(z)+np.exp(-z) ) + 0.5*(c2+c3)
    return res

def fit_mtanh(xdata,ydata,**kwargs):
    fac = 1.
    if ydata.max() > 1e10:  fac = 1e19
    #get initial guesses
    Ldata = ydata/np.abs(np.gradient(ydata,xdata))
    pedloc = xdata[np.argmax(Ldata)]
    pedmin = ydata.min()/fac
    pedmax = ydata.max()/fac
    pedwidth = np.std(Ldata - Ldata.min())
    p0 = [pedloc, pedwidth, pedmax, pedmin, 0.0]
    #fit 
    a,aCovar=curve_fit(mtanh,xdata,ydata/fac, p0 = p0, **kwargs)
    #rescale
    a[2]=a[2]*fac
    a[3]=a[3]*fac
    return a,aCovar



class pfile():
    def __init__(self, pfilename, outfile_prefix = '', write_fits = False):

        self.pfilename = pfilename
        self.outfile_prefix = outfile_prefix
        #read data from pfile
        self.xdata, self.ydata, self.labels = self.read_pfile(self.pfilename)
        #fit pfile data
        self.fits = self.fit_pfile()
        #write out fit profile data for XGC
        if write_fits: self.write_fits()
    

    def read_pfile(self,pfilename):
        labels = []
        xdata = []
        ydata = []
        i = -1; j = 0
        f = open(pfilename)
        for line in f:
            if ('SPECIES' in line):
                N = int(line.strip().split()[0])
                for s in range(N):
                    #for now, dont save out info
                    line = next(f)
            elif ('psinorm' in line):
                N = int(line.strip().split()[0])
                i += 1
                xdata += [np.empty((N,))]
                ydata += [np.empty((N,))]
                j = 0
                labels += [line.strip().split()[2]]
            else:
                xdata[i][j],ydata[i][j] = np.array(line.strip().split()[0:2]).astype('float')
                j += 1
        return xdata,ydata,labels


    def write_fits(self):
        shotstr,timestr = os.path.basename(self.pfilename).split('.')
        shotstr = shotstr[1:] #remove p
        self.write_profile(self.outfile_prefix+shotstr+'.'+timestr+'_ne.dat',self.fits['psinneOut'],self.fits['neOut']*1e20)
        self.write_profile(self.outfile_prefix+shotstr+'.'+timestr+'_te.dat',self.fits['psinTeOut'],self.fits['TeOut'])
        self.write_profile(self.outfile_prefix+shotstr+'.'+timestr+'_ti.dat',self.fits['psinTiOut'],self.fits['TiOut'])
        for s in range(1,4):
            spstr = 'nz'+str(s)
            if spstr+'Out' in self.fits.keys():
                self.write_profile(self.outfile_prefix+shotstr+'.'+timestr+'_'+spstr+'.dat',self.fits['psin'+spstr+'Out'],self.fits[spstr+'Out']*1e20)
    

    def write_profile(self,filename,x,y):
        f = open(filename,'w')
        f.write('%i \n' % len(x))
        for (xi,yi) in zip(x,y):
            f.write('%2.8e\t%4.8e\n' % (xi,yi))
        f.write('-1')
        f.close()


    def fit_pfile(self):
        #fit ne
        lowbnds = np.zeros((5,))
        upbnds = np.ones((5,))
        #force the base to match min value
        upbnds[3] = self.ydata[0].min()

        ane,anec = fit_mtanh(self.xdata[0][self.xdata[0]>0.7],self.ydata[0][self.xdata[0]>0.7],bounds=(lowbnds,upbnds))
        dp = np.diff(self.xdata[0][-50:]).mean()
        psinSOLne = np.arange(self.xdata[0][-1]+dp,self.xdata[0][-1]+25*dp,dp )
        psinneOut = np.hstack( (self.xdata[0],psinSOLne) )

        #neSolFit = mtanh(psinSOLne,*ane)
        lambdane_psin = 2.9676 * 9e-3 #hardcoded average midplane lambda_ne in psin units (8.6 mm)
        neSolFit = self.ydata[0][-1]*np.exp(-(psinSOLne - self.xdata[0][-1])/lambdane_psin)
        neOut = np.hstack( (self.ydata[0], neSolFit) )


        #fit Te
        lowbnds = np.zeros((5,))
        upbnds = np.ones((5,))
        lowbnds[3] = 10./1e3
        upbnds[3] = 20./1e3

        aTe,aTec = fit_mtanh(self.xdata[1][self.xdata[1]>0.7],self.ydata[1][self.xdata[1]>0.7],bounds=(lowbnds,upbnds))
        dp = np.diff(self.xdata[1][-50:]).mean()
        psinSOLTe = np.arange(self.xdata[1][-1]+dp,self.xdata[1][-1]+25*dp,dp )
        psinTeOut = np.hstack( (self.xdata[1],psinSOLTe) )

        TeSolFit = mtanh(psinSOLTe,*aTe)
        TeOut = np.hstack( (self.ydata[1], TeSolFit) )
        TeOut = TeOut*1e3 #keV -> eV

        #fit Ti
        lowbnds = np.zeros((5,))
        upbnds = np.ones((5,))
        lowbnds[3] = 10./1e3
        upbnds[3] = 200./1e3

        aTi,aTic = fit_mtanh(self.xdata[3][self.xdata[3]>0.7],self.ydata[3][self.xdata[3]>0.7],bounds=(lowbnds,upbnds))
        dp = np.diff(self.xdata[3][-50:]).mean()
        psinSOLTi = np.arange(self.xdata[3][-1]+dp,self.xdata[3][-1]+25*dp,dp )
        psinTiOut = np.hstack( (self.xdata[3],psinSOLTi) )

        TiSolFit = mtanh(psinSOLTi,*aTi)
        TiOut = np.hstack( (self.ydata[3], TiSolFit) )
        TiOut = TiOut*1e3 #keV -> eV

        results = {'psinneOut':psinneOut, 'neOut': neOut, 
                   'psinTeOut':psinTeOut, 'TeOut': TeOut, 
                   'psinTiOut':psinTiOut, 'TiOut': TiOut 
                  }
        
        #impurities
        for i in range(1,4): 
            indnz = [ind for ind,s, in enumerate(self.labels) if 'nz'+str(i) in s]  
            if indnz:
                dp = np.diff(self.xdata[indnz[0]][-50:]).mean()
                psinSOLnz = np.arange(self.xdata[indnz[0]][-1]+dp,self.xdata[indnz[0]][-1]+25*dp,dp )
                psinnzOut = np.hstack( (self.xdata[indnz[0]],psinSOLnz) )
                nzSolFit = self.ydata[indnz[0]][-1]*np.exp(-(psinSOLnz - self.xdata[indnz[0]][-1])/lambdane_psin)
                nzOut = np.hstack( (self.ydata[indnz[0]], nzSolFit) )
                results['psinnz'+str(i)+'Out'] = psinnzOut
                results['nz'+str(i)+'Out'] = nzOut

        return results
            


    def plot_fits(self):
        plt.figure()
        ax1 = plt.subplot(311)
        plt.plot(self.fits['psinneOut'],self.fits['neOut'],'-o')
        plt.plot(self.xdata[0],self.ydata[0],'r')
        ax2 = plt.subplot(312,sharex=ax1)
        plt.plot(self.fits['psinTeOut'],self.fits['TeOut'],'-o')
        plt.plot(self.xdata[1],self.ydata[1]*1e3,'r')
        plt.subplot(313,sharex=ax1)
        plt.plot(self.fits['psinTiOut'],self.fits['TiOut'],'-o')
        plt.plot(self.xdata[3],self.ydata[3]*1e3,'r')
        plt.xlabel('$\psi_N$')

    
class mesh_xgc():

    def __init__(self,pfilename, gfilename, **kwargs):
        self.pfilename = pfilename
        self.pobj = pfile(pfilename)
        self.eq = OMFITgeqdsk(gfilename)

        self.psinOut = self.pobj.fits['psinneOut']
        self.RmidOut = self.calc_Rmid(self.psinOut)
        #these will be different if psin_max set (to extrapolate)
        self.psin_mesh = self.psinOut.copy()
        self.Rmid_mesh = self.RmidOut.copy()
        self.Lne, self.LTe, self.LTi = self.scale_lengths(self.RmidOut,
                                                    self.pobj.fits['neOut'],
                                                    self.pobj.fits['TeOut'],
                                                    self.pobj.fits['TiOut'])
        self.rhoi = self.calc_rhoi(self.RmidOut, self.Z0, self.pobj.fits['TiOut'])

        self.spacing = self.calc_spacing(**kwargs)

    def calc_spacing(self, fact_Lne=10., fact_LTe=10., fact_LTi=10., min_spacing=0., max_spacing=7.5e-3, min_rhoi_spacing=False, psin_max=None):
        self.fact_Lne=fact_Lne
        self.fact_LTe=fact_LTe
        self.fact_LTi=fact_LTi
        self.min_spacing=min_spacing
        self.max_spacing=max_spacing

        #find min/max
        spacing = np.min( np.vstack( (max_spacing*np.ones(self.psinOut.shape),
                         self.Lne/fact_Lne, self.LTe/fact_LTe, self.LTi/fact_LTi) ),axis=0)
        spacing = np.max( np.vstack( (spacing, min_spacing*np.ones(self.psinOut.shape)) ), axis=0)
        if min_rhoi_spacing:
            spacing = np.max( np.vstack( (spacing, self.rhoi) ), axis=0)
        #make a smoother transition to fine-scale spacing
        spacinghalf = (spacing.max()+spacing.min())/2.
        indhalf = np.where(spacing<spacinghalf)[0][0]
        indstrans = np.where( (self.psinOut>0.5) & (self.psinOut<self.psinOut[indhalf]) )[0]
        spacing[indstrans] = np.linspace(spacing.max(),spacinghalf,indstrans.size)
        #create a min for SOL
        minspace = max(spacing.min(),0.)
        spacingquarter = (spacing.max()-minspace)/4.+minspace
        spacing[spacing<0]=0.
        spacing[(spacing>spacingquarter) & (self.psinOut>1)] = spacingquarter
        if psin_max is not None:
            if psin_max > self.psinOut[-1]:
                #TODO: probably need a check that psin_max < psinWall
                spacing = np.append(spacing,max_spacing)
                self.psin_mesh = np.append(self.psin_mesh,psin_max)
                self.Rmid_mesh = np.append(self.Rmid_mesh,self.calc_Rmid(psin_max))
        return spacing

    def write_spacing(self, fact_dpol = 3, xgca=True):
        #create inter_curve_spacing_file
        psinSurf = [1.0]
        Rsurf = [np.interp(1.0,self.psin_mesh,self.Rmid_mesh)]
        #forward from psin = 1.0 to wall
        while True:
            Rnew = Rsurf[-1]+np.interp(Rsurf[-1],self.Rmid_mesh,self.spacing)
            if (Rnew>self.Rmid_mesh.max()):
                break
            Rsurf += [Rnew]
            psinSurf += [np.interp(Rsurf[-1],self.Rmid_mesh,self.psin_mesh)]

        #backward from psin = 1.0 to core
        while True:
            Rnew = Rsurf[0]-np.interp(Rsurf[0],self.Rmid_mesh,self.spacing)
            if Rnew<=self.R0:
                #make sure 0.0 surface is there
                Rsurf[0] = self.R0
                psinSurf[0] = 0.0
                break
            Rsurf.insert(0,Rnew)
            psinSurf.insert(0,np.interp(Rnew,self.Rmid_mesh,self.psin_mesh))
            
        Rsurf = np.array(Rsurf)
        psinSurf = np.array(psinSurf)
        
        #create filenames
        shotstr,timestr = os.path.basename(self.pfilename).split('.')
        shotstr = shotstr[1:] #remove p
        file_surf = shotstr+'.'+timestr+'_surf.dat'
        file_dpol = shotstr+'.'+timestr+'_dpol.dat'
        #write surf file
        with open(file_surf,'w') as f:
            f.write('%d \n' % len(psinSurf))
            for p in psinSurf:
                f.write('%2.6e \n' % p)
            f.close()
        
        #write dpol file
        dpol = np.gradient(Rsurf) #equal dR and dpol
        if xgca: dpol = fact_dpol*dpol #can be 3-5x for neoclassical
        with open(file_dpol,'w') as f:
            f.write(str(len(psinSurf))+'\n')
            for (p,d) in zip(psinSurf,dpol):
                f.write('%2.6e\t%2.6e\n' % (p,d))
            f.close()


    def plot_spacing(self):
        plt.figure()
        plt.plot(self.psinOut,self.Lne/self.fact_Lne,label='Lne/'+str(self.fact_Lne),color='blue')
        plt.plot(self.psinOut,self.LTe/self.fact_LTe,label='LTe/'+str(self.fact_LTe),color='red')
        plt.plot(self.psinOut,self.LTi/self.fact_LTi,label='LTi/'+str(self.fact_LTi),color='green')
        plt.plot(self.psinOut,self.rhoi,label=r'$\rho_i$',color='black')
        plt.plot(self.psinOut,self.max_spacing*np.ones(self.psinOut.size),color='black',linestyle='--')
        plt.plot(self.psin_mesh,self.spacing,'--',color='orange',linewidth=2)

        plt.xlabel(r'$\psi_N$')
        plt.ylabel('[m]')
        plt.ylim([0,0.05])
        plt.legend()


    def calc_Rmid(self,psin):
        R=self.eq['AuxQuantities']['R']
        Z=self.eq['AuxQuantities']['Z']
        psinRZ=self.eq['AuxQuantities']['PSIRZ_NORM']
        self.R0=self.eq['fluxSurfaces']['R0']
        self.Z0=self.eq['fluxSurfaces']['Z0']
        RmidGrid=np.linspace(self.R0,np.max(R),1000)
        splPsin=interpolate.RectBivariateSpline(R,Z,psinRZ.T)
        psinGrid=np.squeeze(splPsin(RmidGrid,self.Z0))

        splRmid=interpolate.interp1d(psinGrid,RmidGrid,bounds_error=False)
        return splRmid(psin)

    def scale_lengths(self,RmidOut, neOut, TeOut, TiOut):
        Lne=neOut/np.abs(np.gradient(neOut)/np.gradient(RmidOut))
        LTe=TeOut/np.abs(np.gradient(TeOut)/np.gradient(RmidOut))
        LTi=TiOut/np.abs(np.gradient(TiOut)/np.gradient(RmidOut))
        return Lne, LTe, LTi

    def calc_rhoi(self,RmidOut, ZmidOut, TiOut): 
        R=self.eq['AuxQuantities']['R']
        Z=self.eq['AuxQuantities']['Z']
        Bt=self.eq['AuxQuantities']['Bt']
        splBt=interpolate.RectBivariateSpline(R,Z,Bt.T)
        BtFit=np.squeeze(np.abs(splBt(RmidOut,ZmidOut)))
        return np.sqrt(2*1.667e-27*1.609e-19*TiOut)/(1.609e-19*BtFit)
        
