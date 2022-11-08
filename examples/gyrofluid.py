#!/usr/bin python
#NOTE: This code is given as-is, without modifying to run with current xgc.py loaders.
#This code was used in the paper "Pressure balance in a lower collisionality, attached tokamak scrape-off layer" R.M. Churchill, et.al
#It used a custom data readout of the gyroaveraged potential and electric field.
#This script could be modified to use the gyroaveraging matrix now output from XGC, and perform the same analysis.

#Try applying the momentum equation directly from the gyrokinetic equations of motion

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import adios as ad
import sys
sys.path.append('/global/homes/r/rchurchi/python/python_xgc/')
import xgc
sys.path.append('/global/homes/r/rchurchi/xgc_analytics/')
from f0analysis import create_sparse_xgc
from scipy import integrate
import seaborn as sns

plt.style.use('seaborn-paper')
params = {
           'axes.labelsize': 12,
           'axes.titlesize':16,
           'text.fontsize': 12,
           'legend.fontsize': 12,
           'xtick.labelsize': 12,
           'ytick.labelsize': 12,
           'text.usetex': False,
           'figure.figsize': [6.5, 4.5]
           }
import matplotlib as mpl
mpl.rcParams.update(params)

sns.set_palette('Paired',11)


if not 'loader' in locals():
    loader = xgc._load('./',skiponeddiag=True)
loader.loadf0mesh()
RZ = loader.RZ
me,chargee,vspace_vole,volface,vthe = loader.moments_params(0)
mi,chargei,vspace_voli,volfaci,vthi = loader.moments_params(1)
mass = [me,mi]
charge = [chargee,chargei]

Bpol = np.sqrt(np.sum(loader.bfield[:,:2]**2.,axis=1))
B = np.sqrt(np.sum(loader.bfield**2.,axis=1))
BP = loader.bfield[:,2]

fm = ad.file('xgc.mesh.bp')
surf_len = fm['surf_len'][...]
surf_idx = fm['surf_idx'][...]
#solinds = surf_idx[50,:surf_len[50]]

#if not 'loader' in locals(): #running first time
#    fluxinds = [4,14,24,34,44]
#    termsmid = np.zeros((10,len(fluxinds),2))
#ind = 0
#fluxind = fluxinds[ind]
#xptind = np.argmin((Bpol/B)[solinds])
#solinds = solinds[:xptind-1]
#solinds = solinds[30:]
#solinds = np.arange(loader.wall_nodes[fluxind]+20,loader.wall_nodes[fluxind]+187)
#solinds = np.arange(loader.wall_nodes[2],loader.wall_nodes[3]+1)

tind = 3000

#create gradient matrix operators
fg = ad.file('xgc.grad_rz.bp')
basis = fg['basis'][...]
nelement_r = fg['nelement_r'][...]
eindex_r = fg['eindex_r'][...]-1
value_r = fg['value_r'][...]
nelement_z = fg['nelement_z'][...]
eindex_z = fg['eindex_z'][...]-1
value_z = fg['value_z'][...]
badinds = np.where(basis>0)[0]
value_r[badinds,:] = 0.0
value_z[badinds,:] = 0.0
#these are actually in psi-theta direction (where basis==0)
gradr = create_sparse_xgc(nelement_r,eindex_r,value_r)
gradz = create_sparse_xgc(nelement_z,eindex_z,value_z)

#form geometric terms
#NOTE:these are all in (psi,theta,phi) coordinates 
gradB = np.zeros((loader.psin.size,3))
gradB[:,0] = gradr.dot(B)
gradB[:,1] = gradz.dot(B)
bdotgradB = Bpol/B*gradB[:,1]
ff0m = ad.file('xgc.f0.mesh.bp')
gradpsi = ff0m['gradpsi'][...]
absgradpsi = np.sqrt(np.sum(gradpsi**2.,axis=1))
nb_curl_nb = ff0m['nb_curl_nb'][...]
#v_curv = curl(B), v_gradb = B cross grad(B)
v_gradb = ff0m['v_gradb'][...]
v_curv = ff0m['v_curv'][...]
#the radial components do not have gradpsi normalized out, normalize here
v_gradb[:,0] = v_gradb[:,0]/absgradpsi
v_curv[:,0] = v_curv[:,0]/absgradpsi
curl_nb = v_curv/B[:,np.newaxis] + v_gradb/B[:,np.newaxis]**2.
nb_cross_gradBoverB = v_gradb/B[:,np.newaxis]**2.

#compute matrix to convert vector in (psi,theta) to (R,Z)
#(also A converts (R,Z) to (psi,theta))
A = np.empty((gradpsi.shape[0],2,2))
A[:,0,0] = gradpsi[:,0]/absgradpsi
A[:,0,1] = gradpsi[:,1]/absgradpsi
A[:,1,0] = -gradpsi[:,1]/absgradpsi
A[:,1,1] = gradpsi[:,0]/absgradpsi
Ainv = np.linalg.inv(A)


##calculate distance along flux surface
def calc_L(iinds,poloidal=False):
    dpol = np.sqrt(np.sum(np.diff(loader.RZ[iinds,:],axis=0)**2.,axis=1))
    if poloidal:
        L = np.cumsum(np.hstack((0,dpol)))
    else:
        L = np.cumsum(np.hstack((0,dpol*(B/Bpol)[iinds[0:-1]])))
    return L




#this is basically the same
#Lpar = integrate.cumtrapz(np.hstack((0,dpol*(B/Bpol)[solinds[1:]])))
#this is basically the same also. LparallelNew was from calc_L from calc_offdiagonal_continued in m499.
#Lpar = LparallelNew[np.in1d(iinds,solinds)]
#solinds = np.array(iinds)


##create integrator along flux surface

##filter for noisy signals (e.g. time derivative)
from scipy.signal import filtfilt,iirdesign
bir,air = iirdesign(0.05,0.1,0.5,20) 
def smooth_single(data):
    '''smooths the LFS data in the SOL between the divertor and midplane'''
    datas = data.copy()
    #interpolate onto regular grid for interpolation
    Lpar1 = np.arange(0.,Lpar.max(),np.diff(Lpar).min())
    datai = np.interp(Lpar1,Lpar,data)
    #smooth with IIR lowpass filter, with filtfilt for 0-phase response
    dataifilt = filtfilt(bir,air,datai)
    #reinterpolate onto normal data
    datas = np.interp(Lpar,Lpar1,dataifilt)
    return datas

def smooth(data):
    '''smooths the LFS data in the SOL between the divertor and midplane'''
    datas = data.copy()
    #treat separatrix separatrely, since Lparallel not well defined
    #this is not as good filtering, but still OK
    startind = loader.wall_nodes[0]+2
    iinds = range(startind,startind+200)
    datas[iinds] = filtfilt(bir,air,datas[iinds])
    for i in range(2,40,2):
        startind = loader.wall_nodes[i]+2
        iinds = range(startind,startind+200)
        LparallelNew = calc_L(iinds)

        #interpolate onto regular grid for interpolation
        LparallelNew1 = np.arange(0.,LparallelNew.max(),np.diff(LparallelNew).min())
        datai = np.interp(LparallelNew1,LparallelNew,data[iinds])
        #smooth with IIR lowpass filter, with filtfilt for 0-phase response
        dataifilt = filtfilt(bir,air,datai)
        #reinterpolate onto normal data
        datas[iinds] = np.interp(LparallelNew,LparallelNew1,dataifilt)
    return datas

def calc_pressure_balance(fluxind,plot=True):
    
    solinds = np.arange(loader.wall_nodes[fluxind]+2,loader.wall_nodes[fluxind]+187)
    
    ###########factors to change for integration
    #use this for \int dV = 2pi*dpsi*\int dlparallel/B
    Lpar = calc_L(solinds)
    int_fact = B[solinds]
    ppara_fact = B[solinds]
    #use this for \int dV = 2pi*dpsi*\int dlpol/Bpol
    #Lpar = calc_L(solinds,poloidal=True)
    #int_fact = Bpol[solinds]
    #ppara_fact = B[solinds]
    #use this for \int dlparallel
    #Lpar = calc_L(solinds)
    #int_fact = np.ones(Lpar.shape)
    #ppara_fact = np.ones(Lpar.shape)
    #############################################
    
    def integrate_Lpar(Lpar,data):
        """Integrates (quasi-indefinite integral, using cumtrapz) data along Lparallel direction
        Lpar: Parallel distance along flux surface
        data: Data to integrate. Always integrates along axis=0 dimension
        """
        if data.ndim>1:
            intdata = np.empty((data.shape[0]-1,data.shape[1]))
            for i in range(data.shape[1]):
                intdata[:,i] = integrate.cumtrapz(data[:,i]/int_fact,x=Lpar)
                #intdata[:,i] = (np.cumsum(data[:,i]*loader.node_vol[solinds]))[1:]#/np.cumsum(loader.node_vol[solinds]))[1:]
        else:
            #intdata = integrate.cumtrapz(data/B[solinds],x=Lpar)
            intdata = integrate.cumtrapz(data/int_fact,x=Lpar)
        return intdata
    
    #########terms in parallel momentum balance
    ##read f2d data
    den = np.zeros((loader.psin.size,2))
    Tpara = np.zeros((loader.psin.size,2))
    Tperp = np.zeros((loader.psin.size,2))
    upara = np.zeros((loader.psin.size,2))

    ff2d = ad.file('xgc.f2d.'+str(tind).zfill(5)+'.bp')
    for (isp,sp) in enumerate(['e','i']):
        den[:,isp] = ff2d[sp+'_den'][...]
        upara[:,isp] = ff2d[sp+'_u_para'][...]
        Tpara[:,isp] = ff2d[sp+'_T_para'][...]
        Tperp[:,isp] = ff2d[sp+'_T_perp'][...]
    ff2d.close()
    ppara = den*Tpara*chargei
    pparatot = ppara + np.einsum('j,ij->ij',mass,den)*upara**2.
    pperp = den*Tperp*chargei
    ptot = np.sum((ppara+2.*pperp)/3.,axis=1) + mi*den[:,1]*upara[:,1]**2.+me*den[:,0]*upara[:,0]**2.



    #create an "effective" Lpar, to account for near 0 parallel motion near X-point
    ##read in gyro-averaged electric field
    fef = ad.file('xgc.e_gyro_avg.'+str(tind).zfill(5)+'.bp')
    E_rho = fef['E_rho'][...]
    pot_rho = fef['pot_rho'][...]
    uexb = E_rho[:,0,0]*BP/B**2.
    umag = np.empty(den.shape)
    umag[:,0] = (v_gradb[:,1]/chargee/B**3.*(pperp[:,0] + ppara[:,0] + me*den[:,0]*upara[:,0]**2.) + 
            v_curv[:,1]/chargee/B**2.*(ppara[:,0]+me*den[:,0]*upara[:,0]**2.))/den[:,0]
    umag[:,1] = (v_gradb[:,1]/chargei/B**3.*(pperp[:,1] + ppara[:,1] + mi*den[:,1]*upara[:,1]**2.) + 
            v_curv[:,1]/chargei/B**2.*(ppara[:,1]+mi*den[:,1]*upara[:,1]**2.))/den[:,1]
    #collisionless, adiabatic sound speed, see Stangeby
    csi = np.sqrt((Tpara[:,0]+3.*Tpara[:,1])*chargei/mi)

    LparOrig = Lpar.copy()
    #dpol = np.sqrt(np.sum(np.diff(loader.RZ[solinds,:],axis=0)**2.,axis=1))
    #fac = B/Bpol/(1.+B/Bpol*uexb*B**2./csi)
    fac = B/Bpol/(1.+B/Bpol*uexb/csi)
    #Lpar = np.cumsum(np.hstack((0,dpol*fac[solinds[0:-1]])))
    #Lpar = Lpar/(1 + (B/Bpol*(uexb+umag[:,1])*B**2./csi)[solinds])


    #term1: time-derivative of m*n*upara
    term1_ddt_mnupara = np.empty((loader.psin.size,2))
    term6_ddt_ppara = np.empty((loader.psin.size,2))
    ff2dp1 = ad.file('xgc.f2d.'+str(tind+1).zfill(5)+'.bp')
    ff2dm1 = ad.file('xgc.f2d.'+str(tind-1).zfill(5)+'.bp')
    steps = np.arange(5).astype(int)-2
    stencil = np.array([1,-8,0,8,-1])/12.
    #steps = np.arange(3).astype(int)-1
    #stencil = np.array([-1,0,1])/2.
    #read in many timeslice to get accurate smoothing
    Ntimepts = 6
    tmp_data1 = np.zeros((loader.psin.size,Ntimepts,2))
    tmp_data6 = np.zeros((loader.psin.size,Ntimepts,2))
    for it in range(Ntimepts):
        ff2d = ad.file('xgc.f2d.'+str(tind+it-Ntimepts/2).zfill(5)+'.bp')
        for (isp,sp) in enumerate(['e','i']):
            dentmp = ff2d[sp+'_den'][...]
            uparatmp = ff2d[sp+'_u_para'][...]
            Tparatmp = ff2d[sp+'_T_para'][...]
            tmp_data1[:,it,isp] = dentmp*uparatmp
            tmp_data6[:,it,isp] = dentmp*Tparatmp*charge[1] + mass[isp]*dentmp*uparatmp**2.
        ff2d.close()
    #now smooth the data
    #tmp_data1[solinds,...] = filtfilt(bir,air,tmp_data1[solinds,...],axis=1)
    #tmp_data6[solinds,...] = filtfilt(bir,air,tmp_data6[solinds,...],axis=1)

    steps = np.arange(5).astype(int) - 2 + Ntimepts/2
    for (isp,sp) in enumerate(['e','i']):
        tmp_sum1 = 0.
        tmp_sum6 = 0.
        for (it,s) in zip(steps,stencil):
            tmp_sum1 += s*tmp_data1[:,it,isp]
            tmp_sum6 += s*tmp_data6[:,it,isp]
        term1_ddt_mnupara[:,isp] = mass[isp]/loader.unit_dic['sml_dt']*tmp_sum1
        term6_ddt_ppara[:,isp] = mass[isp]/charge[isp]/B/loader.unit_dic['sml_dt']*tmp_sum6
    int_term1 = integrate_Lpar(Lpar,term1_ddt_mnupara[solinds,:]) 
    int_term6 = integrate_Lpar(Lpar,term6_ddt_ppara[solinds,:]) 

    #term2: parallel pressure gradient
    term2_gradppara = np.empty((loader.psin.size,2))
    int_term2 = np.empty((solinds[1:].size,2))
    for (isp,sp) in enumerate(['e','i']):
        term2_gradppara[:,isp] = Bpol/B*gradz.dot(pparatot[:,isp])
        int_term2[:,isp] = pparatot[solinds[:-1],isp]
    #OPTIONAL: integration of gradient instead. Add in x=0, so like int_term2 above
    int_term2 = integrate_Lpar(Lpar,term2_gradppara[solinds,:])
    int_normalize = pparatot[solinds[0],:]/ppara_fact[0]
    #int_normalize = pparatot[solinds[0],:]
    #int_normalize = pparatot[solinds[0],:]*(Bpol/B)[solinds[0]]
    #int_normalize = int_term2[0,:]
    int_term2[:,:] += int_normalize

    #int_term2[:,:] += pparatot[solinds[0],:]/B[solinds[0]]
    #int_term2[:,:] += pparatot[solinds[0],:]
    #int_term2[:,:] += int_term2[0,:]

    #term3: perpendicular pressure
    term3_visc = np.empty((loader.psin.size,2))
    term3_visc[:,0] = (bdotgradB/B)*(pperp[:,0]-ppara[:,0]-me*den[:,0]*upara[:,0]**2.)
    term3_visc[:,1] = (bdotgradB/B)*(pperp[:,1]-ppara[:,1]-mi*den[:,1]*upara[:,1]**2.)
    int_term3 = integrate_Lpar(Lpar,term3_visc[solinds,:])

    ##form gyro-averaged electric field
    #create gyro-average matrix (these were written out by separate simulation)
    nrho = 6
    rhomax = 1e-2
    drho = rhomax/nrho
    #def read_gyro_avg_mat(filename):
    #    f = ad.file(filename)
    #    nelement = f['nelement'][...]
    #    eindex = f['eindex'][...]-1
    #    value = f['value'][...]
    #    return create_sparse_xgc(nelement,eindex,value) 
    #gyro_avg_mat = []
    #for i in range(nrho):
    #    gyro_avg_mat.append(read_gyro_avg_mat('../xgca_gyro_avg_mat/xgc.gyro_avg_mat.'+str(i).zfill(5)+'.bp'))
    #
    ##read in potential (off by half-time step :( )
    #f2d = ad.file('xgc.2d.'+str(tind).zfill(5)+'.bp')
    #pot = f2d['pot0'][...]+f2d['dpot'][...]
    ##form gyro-averaged Efield
    #E_rho = np.zeros((loader.psin.size,3,len(gyro_avg_mat)+1))
    #E_rho[:,0,0],E_rho[:,1,0] = -gradr.dot(pot),-gradz.dot(pot)
    #for i in range(6):
    #    E_rho[:,0,i+1] = gyro_avg_mat[i].dot(E_rho[:,0,0])
    #    E_rho[:,1,i+1] = gyro_avg_mat[i].dot(E_rho[:,1,0])


    #we need to interpolate the E_rho grid onto the f0_f grid.
    #the f0_f grid is normalized at each node, the E_rho grid
    #is NOT, its constant throughout
    #shape (Nvpe,Nnode)
    rhoi = mi/chargei * np.einsum('i,j->ij',loader.vpe,vthi/B)

    rhog = np.arange(nrho+1)*drho
    rhoilim = np.minimum(rhoi,rhomax)
    E_rhog = np.zeros(rhoilim.shape+(2,))
    pot_rhog = np.zeros(rhoilim.shape)
    for i in range(rhoilim.shape[1]):
        E_rhog[:,i,0] = np.interp(rhoilim[:,i],rhog,E_rho[i,:,0])
        E_rhog[:,i,1] = np.interp(rhoilim[:,i],rhog,E_rho[i,:,1])
        pot_rhog[:,i] = np.interp(rhoilim[:,i],rhog,pot_rho[i,:])

    ff0 = ad.file('xgc.f0.'+str(tind).zfill(5)+'.bp')
    i_f = ff0['i_f'][...]
    e_f = ff0['e_f'][...]
    #term4: q<Epar>
    term4_qEpar = np.empty((loader.psin.size,2))
    term4_qEpar[:,1] = -Bpol/B*np.einsum('ij,ijk,ik->j',chargei*E_rhog[:,:,1],i_f,volfaci)*vspace_voli
    term4_qEpar[:,0] = -Bpol/B*chargee*den[:,0]*E_rho[:,0,1]
    #int_term4[:,0] = chargee*den[solinds[:-1],0]*pot_rho[solinds[:-1],0]
    #this is wrong, we dont form grad(\bar{Phi}), we form \bar{grad(Phi)} (bar is gyro-average)
    #int_term4all = np.einsum('ij,ijk,ik->j',chargei*pot_rhog,i_f,volfaci)*vspace_voli
    #int_term4[:,1] = int_term4all[solinds[:-1]]
    #int_term4 = int_term4 - int_term4[0,:]
    int_term4 = integrate_Lpar(Lpar,term4_qEpar[solinds,:])

    ##read in source distribution function, calculate moments
    #these source are df=S*dt. To get S*, divide by dt
    e_f_source = ff0['e_f_source'][...]/loader.unit_dic['sml_dt']
    i_f_source = ff0['i_f_source'][...]/loader.unit_dic['sml_dt']
    ff0.close()
    seden,seupara,_,seTpara,seTperp = loader.calcMoments1(e_f_source,0)
    siden,siupara,_,siTpara,siTperp = loader.calcMoments1(i_f_source,1)
    seppara = seden*seTpara*chargei
    sippara = siden*siTpara*chargei
    #term5: <mvpara>_source
    term5_source = np.empty((loader.psin.size,2))
    #int_term5 = np.empty((solinds[1:].size,2))
    term5_source[:,0] = -me*seden*seupara
    term5_source[:,1] = -mi*siden*siupara
    int_term5 = integrate_Lpar(Lpar,term5_source[solinds,:])
    term13_source = np.empty((loader.psin.size,2))
    term13_source[:,0] = -me/chargee/B*nb_curl_nb*(seppara + me*seden*seupara**2.)
    term13_source[:,1] = -mi/chargei/B*nb_curl_nb*(sippara + mi*siden*siupara**2.)
    int_term13 = integrate_Lpar(Lpar,term13_source[solinds,:])

    ##m/qB*curl(b) terms
    mVpaCube = me*vthe**3.*np.einsum('k,ijk,ik->j',loader.vpa**3.,e_f,volface)*vspace_vole
    mVpaCubi = mi*vthi**3.*np.einsum('k,ijk,ik->j',loader.vpa**3.,i_f,volfaci)*vspace_voli
    term7_vpar3 = np.empty((loader.psin.size,2))
    term7_vpar3[:,0] = me/chargee/B*(curl_nb[:,0]*gradr.dot(mVpaCube) + curl_nb[:,1]*gradz.dot(mVpaCube))
    term7_vpar3[:,1] = mi/chargei/B*(curl_nb[:,0]*gradr.dot(mVpaCubi) + curl_nb[:,1]*gradz.dot(mVpaCubi))
    int_term7 = integrate_Lpar(Lpar,term7_vpar3[solinds,:])

    term8_vpar3 = np.empty((loader.psin.size,2))
    term8_vpar3[:,0] = -me/chargee/B*(curl_nb[:,0]*gradB[:,0]/B*mVpaCube + curl_nb[:,1]*gradB[:,1]/B*mVpaCube)
    term8_vpar3[:,1] = -mi/chargei/B*(curl_nb[:,0]*gradB[:,0]/B*mVpaCubi + curl_nb[:,1]*gradB[:,1]/B*mVpaCubi)
    int_term8 = integrate_Lpar(Lpar,term8_vpar3[solinds,:])

    mVpaVpe2e = me*vthe**3.*np.einsum('k,i,ijk,ik->j',loader.vpa,loader.vpe**2.,e_f,volface)*vspace_vole
    mVpaVpe2i = mi*vthi**3.*np.einsum('k,i,ijk,ik->j',loader.vpa,loader.vpe**2.,i_f,volfaci)*vspace_voli
    term9_vparvperp2 = np.empty((loader.psin.size,2))
    term9_vparvperp2[:,0] = me/chargee/B*(curl_nb[:,0]*gradB[:,0]/B*mVpaVpe2e + curl_nb[:,1]*gradB[:,1]/B*mVpaVpe2e)
    term9_vparvperp2[:,1] = mi/chargei/B*(curl_nb[:,0]*gradB[:,0]/B*mVpaVpe2i + curl_nb[:,1]*gradB[:,1]/B*mVpaVpe2i)
    int_term9 = integrate_Lpar(Lpar,term9_vparvperp2[solinds,:])

    VpaqEi = chargei*vthi[:,np.newaxis]*np.einsum('k,ijl,ijk,ik->jl',loader.vpa,E_rhog,i_f,volfaci)*vspace_voli[:,np.newaxis]
    VpaqEe = chargee*(den[:,0]*upara[:,0])[:,np.newaxis]*E_rho[:,0,0:2]
    term10_vparaE = np.empty((loader.psin.size,2))
    term10_vparaE[:,0] = me/chargee/B*-2.*(curl_nb[:,0]*VpaqEe[:,0]+curl_nb[:,1]*VpaqEe[:,1])
    term10_vparaE[:,1] = mi/chargei/B*-2.*(curl_nb[:,0]*VpaqEi[:,0]+curl_nb[:,1]*VpaqEi[:,1])
    int_term10 = integrate_Lpar(Lpar,term10_vparaE[solinds,:])


    ##m/qB nb_cross_gradBoverB
    term11_vpavpe2 = np.empty((loader.psin.size,2))
    term11_vpavpe2[:,0] = me/chargee/B*(nb_cross_gradBoverB[:,0]*gradr.dot(0.5*mVpaVpe2e) + nb_cross_gradBoverB[:,1]*gradz.dot(0.5*mVpaVpe2e))
    term11_vpavpe2[:,1] = mi/chargei/B*(nb_cross_gradBoverB[:,0]*gradr.dot(0.5*mVpaVpe2i) + nb_cross_gradBoverB[:,1]*gradz.dot(0.5*mVpaVpe2i))
    int_term11 = integrate_Lpar(Lpar,term11_vpavpe2[solinds,:])

    #to calculate bhat \cdot curl(<vpa*q*E>), I have to do the curl first.
    #follow the formula used for v_curv (which calculates curl(B))
    #For an axisymmetric vector, with no toroidal component, curl(G) = phihat (dGR/dZ - dGZ/dR) 
    #since VpaqEi is in (psi,theta) coordinates, I first have to rotate the vector to (R,Z),
    #then form dGi/dLpsi,dGi/dLtheta (using gradr,gradz), then rotate those also to (R,Z)
    term12_vpaqE = np.empty((loader.psin.size,2))
    for (isp,sp) in enumerate(['e','i']):
        if isp==0: 
            VpaqE = VpaqEe.copy() 
        else: 
            VpaqE = VpaqEi.copy()
        VpaqERZ = np.einsum('ijk,ik->ij',Ainv,VpaqE)
        alphaPsi,alphaTheta = gradr.dot(VpaqERZ[:,0]),gradz.dot(VpaqERZ[:,0])
        betaPsi,betaTheta = gradr.dot(VpaqERZ[:,1]),gradz.dot(VpaqERZ[:,1])
        alphaRZ = np.einsum('ijk,ik->ij',Ainv,np.array([alphaPsi,alphaTheta]).T)
        betaRZ = np.einsum('ijk,ik->ij',Ainv,np.array([betaPsi,betaTheta]).T)
        term12_vpaqE[:,isp] = mass[isp]/charge[isp]/B*BP/B*(alphaRZ[:,1] - betaRZ[:,0])
    int_term12 = integrate_Lpar(Lpar,term12_vpaqE[solinds,:])

    termsmid = np.zeros((10,2))
    termsmid[0,:] = (int_term2[-1,:]-int_term2[0,:])/int_normalize
    termsmid[1,:] = (int_term1[-1,:]+int_term6[-1,:])/int_normalize
    termsmid[2,:] = int_term3[-1,:]/int_normalize
    termsmid[3,:] = int_term4[-1,:]/int_normalize
    termsmid[4,:] = (int_term5[-1,:]+int_term13[-1,:])/int_normalize
    termsmid[5,:] = int_term7[-1,:]/int_normalize
    termsmid[6,:] = (int_term8[-1,:]+int_term9[-1,:])/int_normalize
    termsmid[7,:] = int_term10[-1,:]/int_normalize
    termsmid[8,:] = int_term11[-1,:]/int_normalize
    termsmid[9,:] = int_term12[-1,:]/int_normalize

    if plot:
        ##plotting
        for (isp,sp) in enumerate(['e','i']):
            plt.figure()
            plt.plot(RZ[solinds[1:],1],(int_term1[:,isp]+int_term6[:,isp]),label='$\partial_t$')
            plt.plot(RZ[solinds[1:],1],int_term2[:,isp],label='$p_\parallel + mnu_\parallel^2$')
            plt.plot(RZ[solinds[1:],1],int_term3[:,isp],label='$p_\parallel - p_\perp$')
            plt.plot(RZ[solinds[1:],1],int_term4[:,isp],label='$E_\parallel$')
            plt.plot(RZ[solinds[1:],1],(int_term5[:,isp]+int_term13[:,isp]),label='source')
            #plt.plot(RZ[solinds[1:],1],int_term6[:,isp],label='$\partial_t p_\parallel + mnu_\parallel^2$')
            plt.plot(RZ[solinds[1:],1],int_term7[:,isp],label='$curl(b) \cdot <mv_\parallel^3>$')
            plt.plot(RZ[solinds[1:],1],int_term8[:,isp],label='$curl(b) \cdot \\nabla B/B <mv_\parallel^3>$')
            plt.plot(RZ[solinds[1:],1],int_term9[:,isp],label='$curl(b) \cdot \\nabla B/B <mv_\parallel v_perp^2>$')
            plt.plot(RZ[solinds[1:],1],int_term10[:,isp],label='$curl(b) \cdot -2<v_\parallel q E >$')
            plt.plot(RZ[solinds[1:],1],int_term11[:,isp],label='$b \\times \\nabla B/B \cdot <0.5mv_\parallel v_perp^3>$')
            plt.plot(RZ[solinds[1:],1],int_term12[:,isp],label='$b \\times  \cdot \\nabla \\times <v_\parallel q E}>$')
            #plt.plot(RZ[solinds[1:],1],int_term13[:,isp],label='source 2')
            plt.title('Raw quatities')
            plt.legend(loc='best')

        for (isp,sp) in enumerate(['e','i']):
            plt.figure()
            plt.plot(RZ[solinds[1:],1],(int_term2[:,isp]-int_term2[0,isp])/int_normalize[isp],label='$(p_\parallel + mnu_\parallel^2)|_0^x$',markevery=5,linewidth=4)
            plt.plot(RZ[solinds[1:],1],(int_term1[:,isp]+int_term6[:,isp])/int_normalize[isp],label='$F_{\partial_t}$',markevery=5)
            plt.plot(RZ[solinds[1:],1],int_term3[:,isp]/int_normalize[isp],label='$F_{visc}$',markevery=5)
            plt.plot(RZ[solinds[1:],1],int_term4[:,isp]/int_normalize[isp],label='$F_{E_\parallel}$',markevery=5)
            plt.plot(RZ[solinds[1:],1],(int_term5[:,isp]+int_term13[:,isp])/int_normalize[isp],label='$F_{source}$',markevery=5)
            #plt.plot(RZ[solinds[1:],1],int_term6[:,isp]/int_normalize[0,isp],label='$\partial_t p_\parallel + mnu_\parallel^2$')
            plt.plot(RZ[solinds[1:],1],int_term7[:,isp]/int_normalize[isp],'-o',label='$F_{\\nabla <mv_\parallel^3>}$',markevery=5)
            plt.plot(RZ[solinds[1:],1],(int_term8[:,isp]+int_term9[:,isp])/int_normalize[isp],'-o',label='$F_{(\\nabla \\times \hat{\mathbf{b}}) \cdot \\nabla B/B}$',markevery=5)
            #plt.plot(RZ[solinds[1:],1],int_term9[:,isp]/int_normalize[0,isp],'-o',label='$(\\nabla \\times b) \cdot \\nabla B/B <mv_\parallel v_\perp^2>$',markevery=5)
            plt.plot(RZ[solinds[1:],1],int_term10[:,isp]/int_normalize[isp],'-o',label='$F_{2<v_\parallel q E >}$',markevery=5)
            plt.plot(RZ[solinds[1:],1],int_term11[:,isp]/int_normalize[isp],'-o',label='$F_{<0.5mv_\parallel v_\perp^2>}$',markevery=5)
            plt.plot(RZ[solinds[1:],1],int_term12[:,isp]/int_normalize[isp],'-o',label='$F_{\hat{\mathbf{b}} \cdot \\nabla \\times <v_\parallel q E>}$',markevery=5)
           # plt.plot(RZ[solinds[1:],1],int_term13[:,isp]/int_normalize[0,isp],label='source 2')
            plt.title('Normalized quatities')
            plt.legend(loc='best')
            plt.xlabel('Z[m]')

        labels = ['$\\frac{p_{\parallel,tot}|_{\ell_\parallel=x}}{p_{\parallel,tot}|_{\ell_\parallel=0}}-1$',
                 '$F_{\partial_t}$',
                 '$F_{visc}$',
                 '$F_{E_\parallel}$',
                 '$F_{source}$',
                 '$F_{\\nabla <mv_\parallel^3>}$',
                 '$F_{(\\nabla \\times \hat{\mathbf{b}}) \cdot \\nabla B/B}$',
                 '$F_{2<v_\parallel q E >}$',
                 '$F_{<0.5mv_\parallel v_\perp^2>}$',
                 '$F_{\hat{\mathbf{b}} \cdot \\nabla \\times <v_\parallel q E>}$']

        totale = (int_term1[:,0] + int_term2[:,0] + 
                  int_term3[:,0] + int_term4[:,0] + 
                  int_term5[:,0]+int_term6[:,0]+
                  int_term7[:,0]+int_term8[:,0]+int_term9[:,0]+int_term10[:,0]+
                  int_term11[:,0]+int_term12[:,0]+int_term13[:,0])/int_normalize[0]#(pparatot[solinds[0],0]/B[solinds[0]])#int_term2[0,0]
        plt.figure()
        plt.plot(RZ[solinds[1:],1],totale)

        totali = (int_term1[:,1] + int_term2[:,1] + 
                  int_term3[:,1] + int_term4[:,1] + 
                  int_term5[:,1]+int_term6[:,1]+
                  int_term7[:,1]+int_term8[:,1]+int_term9[:,1]+int_term10[:,1]+
                  int_term11[:,1]+int_term12[:,1]+int_term13[:,1])/int_normalize[1]#(pparatot[solinds[0],1]/B[solinds[0]])#int_term2[0,1]
        plt.figure()
        plt.plot(RZ[solinds[1:],1],totali)

        total = (int_normalize[0]*totale + int_normalize[1]*totali)/(int_normalize[0]+int_normalize[1])
        plt.figure()
        plt.plot(RZ[solinds[1:],1],total,'-o',label='Gyrofluid',markevery=5)
        plt.plot(RZ[solinds[1:],1],ptot[solinds[1:]]/ptot[solinds[1]],'-d',label='$p_{tot}$',markevery=5)
        plt.xlabel('Z[m]')
        plt.legend(loc='best',fontsize=16)
        plt.tight_layout()
    
    return termsmid

fluxinds = np.arange(4,44,2)
termsmid = np.zeros((10,len(fluxinds),2))
for (i,fluxind) in enumerate(fluxinds):
    print i,fluxind
    termsmid[:,i,:] = calc_pressure_balance(fluxind,plot=False)

#run plot_gyrofluid_fill 
