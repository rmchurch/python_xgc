#!/usr/bin python

import time
import adios2 as ad

def read_fluc_single(i,xgc_path,rzInds,phi_start,phi_end):
    f = ad.open(xgc_path + 'xgc.3d.'+str(i).zfill(5)+'.bp','r')
    inds1d = (rzInds,)
    inds2d=(slice(phi_start,phi_end+1),)+inds1d
    dpot1 = f.read('dpot')[inds2d]
    pot01 = f.read('pot0')[inds1d]
    eden1 = f.read('eden')[inds2d]
    f.close()
    print(i)
    return i,dpot1,pot01,eden1

def read_fluc_single_tmp(i,readCmd,xgc_path,rzInds,phi_start,phi_end):
    flucFile = ad.open(xgc_path + 'xgc.3d.'+str(i).zfill(5)+'.bp','r')
    dpot1 = readCmd(flucFile,'dpot',inds=(slice(phi_start,phi_end+1),)+(rzInds,) )#[self.rzInds,self.phi_start:(self.phi_end+1)]
    pot01 = readCmd(flucFile,'pot0',inds=(rzInds,) )#[rzInds]
    eden1 = readCmd(flucFile,'eden',inds=(slice(phi_start,phi_end+1),)+(rzInds,) )#[self.rzInds,self.phi_start:(self.phi_end+1)]
    flucFile.close()
    return i,dpot1,pot01,eden1
