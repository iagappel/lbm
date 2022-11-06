#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, time
import subprocess

import h5py
import pickle

import math
import numpy as np
import scipy
from scipy import ndimage, signal

import tqdm
from tqdm import tqdm
import shutil
import psutil
import pathlib
from pathlib import Path, PurePosixPath

import numba
from numba import jit,njit

import datetime
import timeit

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cmocean, colorcet, cmasher

import PIL

'''
========================================================================

Lattice-Boltzmann Method (LBM) Simulator

A 2D D2Q9 implementation of the Lattice-Boltzmann Method

-----

$> NUMBA_NUM_THREADS=8 python3 lbm.py

========================================================================
'''

# utilities
# ======================================================================

def format_time_string(tsec):
    '''
    format seconds as dd:hh:mm:ss
    '''
    m, s = divmod(tsec,60)
    h, m = divmod(m,60)
    d, h = divmod(h,24)
    time_str = '%dd:%dh:%02dm:%02ds'%(d,h,m,s)
    return time_str

def even_print(label, output, **kwargs):
    '''
    print/return a fixed width message
    '''
    terminal_width = kwargs.get('terminal_width',72)
    s              = kwargs.get('s',False) ## return string
    
    ndots = (terminal_width-2) - len(label) - len(output)
    text = label+' '+ndots*'.'+' '+output
    if s:
        return text
    else:
        #sys.stdout.write(text)
        print(text)
        return

# Lattice Boltzmann Method (LBM)
# ======================================================================

@njit(parallel=True, fastmath=True)
def equilibrium(rho, ux, uy):
    '''
    get equilibrium distribution from macroscopic variables [u,v,ρ]
    '''
    ux2   = ux**2
    uy2   = uy**2
    umag2 = ux2+uy2
    d = 9/2
    e = (3/2)*umag2
    ##
    a = 4/9  ## 'rest' omega
    b = 1/9  ## 'slow' omega
    c = 1/36 ## 'fast' omega
    ##
    feq_n0  = rho*a*(1 + (                                     - e))
    feq_nN  = rho*b*(1 + ( 3*uy      + d*uy2                   - e))
    feq_nS  = rho*b*(1 + (-3*uy      + d*uy2                   - e))
    feq_nE  = rho*b*(1 + ( 3*ux      + d*ux2                   - e))
    feq_nW  = rho*b*(1 + (-3*ux      + d*ux2                   - e))
    feq_nNE = rho*c*(1 + (3*( ux+uy) + d*(ux2 + 2*ux*uy + uy2) - e))
    feq_nNW = rho*c*(1 + (3*(-ux+uy) + d*(ux2 - 2*ux*uy + uy2) - e))
    feq_nSE = rho*c*(1 + (3*( ux-uy) + d*(ux2 - 2*ux*uy + uy2) - e))
    feq_nSW = rho*c*(1 + (3*(-ux-uy) + d*(ux2 + 2*ux*uy + uy2) - e))
    
    feq = np.stack((feq_n0,feq_nN,feq_nS,feq_nE,feq_nW,feq_nNE,feq_nNW,feq_nSE,feq_nSW), axis=0)
    
    return feq

@njit(parallel=True, fastmath=True)
def collide(f_streamed,tau):
    '''
    collision and relaxation of the distribution function
    '''
    
    # rho, ux, uy = get_macros(f_streamed)
    ##
    n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW = f_streamed
    rho = n0 + nN + nS + nE + nW + nNE + nSE + nNW + nSW
    ux  = (nE + nNE + nSE - nW - nNW - nSW) / rho
    uy  = (nN + nNE + nNW - nS - nSE - nSW) / rho
    ##
    feq = equilibrium(rho, ux, uy)
    ##
    f = f_streamed - (1/tau)*(f_streamed-feq) ## relaxation
    
    return f, rho, ux, uy

@njit
def get_macros(f):
    '''
    get macroscopic variables [u,v,ρ] from the distribution f
    '''
    n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW = f
    rho = n0 + nN + nS + nE + nW + nNE + nSE + nNW + nSW
    ux = (nE + nNE + nSE - nW - nNW - nSW) / rho
    uy = (nN + nNE + nNW - nS - nSE - nSW) / rho
    #p = rho * cs**2
    return rho, ux, uy

@njit(parallel=True, fastmath=True)
def stream(f,obstacle):
    '''
    streaming step (advect the distribution function)
    '''
    
    n_directions,nx,ny = f.shape
    f_streamed = np.zeros((n_directions,nx,ny), dtype=np.float64)
    
    i0, iN, iS, iE, iW, iNE, iNW, iSE, iSW = 0,1,2,3,4,5,6,7,8
    
    ## stream
    for i in range(nx):
        for j in range(ny):
            
            # ============================================================ #
            # Set 'wrap around' conditions
            # - periodicity is the 'default' boundary condition
            # - explicit boundary conditions (like an inlet) can be set
            #    immediately afterward which overwrite the distribution
            #    function at a selected boundary.
            # ============================================================ #
            
            if (i==0):
                i_minus = nx-1
            else:
                i_minus = i-1
            
            if (i==nx-1):
                i_plus = 0
            else:
                i_plus = i+1
            
            if (j==0):
                j_minus = ny-1
            else:
                j_minus = j-1
            
            if (j==ny-1):
                j_plus = 0
            else:
                j_plus = j+1
            
            # ============================================================ #
            # Advection
            # ---------
            # - fluxes  N get transported to [ 0,+1]
            # - fluxes  W get transported to [-1, 0]
            # - fluxes NW get transported to [-1,+1]
            # ...
            # - fluxes  0 get dont get transported
            # ============================================================ #
            
            f_streamed[ i0  , i , j ] = f[ i0  , i        , j       ]
            f_streamed[ iW  , i , j ] = f[ iW  , i_plus   , j       ]
            f_streamed[ iE  , i , j ] = f[ iE  , i_minus  , j       ]
            f_streamed[ iS  , i , j ] = f[ iS  , i        , j_plus  ]
            f_streamed[ iN  , i , j ] = f[ iN  , i        , j_minus ]
            f_streamed[ iSW , i , j ] = f[ iSW , i_plus   , j_plus  ]
            f_streamed[ iNW , i , j ] = f[ iNW , i_plus   , j_minus ]
            f_streamed[ iSE , i , j ] = f[ iSE , i_minus  , j_plus  ]
            f_streamed[ iNE , i , j ] = f[ iNE , i_minus  , j_minus ]
    
    f_streamed2 = np.zeros((n_directions,nx,ny), dtype=np.float64)
    f_streamed2[:,:,:] = f_streamed[:,:,:]
    
    ## bounce back (from obstacle)
    for i in range(nx):
        for j in range(ny):
            if (i!=0) and (j!=0) and (i!=nx-1) and (j!=ny-1): ## non-boundary points
                ##
                if obstacle[i,j+1]:
                    f_streamed[iS,i,j] = f_streamed2[iN,i,j+1]
                if obstacle[i,j-1]:
                    f_streamed[iN,i,j] = f_streamed2[iS,i,j-1]
                if obstacle[i+1,j]:
                    f_streamed[iW,i,j] = f_streamed2[iE,i+1,j]
                if obstacle[i-1,j]:
                    f_streamed[iE,i,j] = f_streamed2[iW,i-1,j]
                ##
                if obstacle[i+1,j+1]:
                    f_streamed[iSW,i,j] = f_streamed2[iNE,i+1,j+1]
                if obstacle[i+1,j-1]:
                    f_streamed[iNW,i,j] = f_streamed2[iSE,i+1,j-1]
                if obstacle[i-1,j+1]:
                    f_streamed[iSE,i,j] = f_streamed2[iNW,i-1,j+1]
                if obstacle[i-1,j-1]:
                    f_streamed[iNE,i,j] = f_streamed2[iSW,i-1,j-1]
    
    return f_streamed

class lbm(object):
    '''
    A Lattice-Boltzmann Method (LBM) Simulator
    -----
    2D, D2Q9 implementation
    '''
    
    def __init__(self, Re=1000, M_inf=0.1, fn_h5='data.h5', nx=854, ny=480, **kwargs):
        
        self.geom_preset      = kwargs.get('geom_preset',None)
        self.obstacle_bmp     = kwargs.get('obstacle_bmp',None) ## point to a .bmp obstacle file
        self.fn_restart_dat   = kwargs.get('restart',None)
        self.max_time_s       = kwargs.get('max_time_s',None) ## max wall clock time
        self.data_save_period = kwargs.get('data_save_period',100)
        self.fps              = kwargs.get('fps',60)
        ##
        self.bcNS = kwargs.get('bcNS','periodic') ## South/North boundary BC
        
        print('\n'+'lbm.__init__()'+'\n'+72*'-')
        self.start_time = timeit.default_timer()
        
        if (self.fn_restart_dat is not None):
            self.start_from_restart = True
        else:
            self.start_from_restart = False
        
        if not isinstance(self.data_save_period,int):
            raise ValueError('data save period must be of type int')
        
        cs = np.sqrt(1/3) ## speed of sound in lattice units
        
        ## load restart
        if self.start_from_restart:
            even_print('restart', 'True (%s)'%(self.fn_restart_dat,))
            self.load_restart(self.fn_restart_dat)
        else:
            even_print('restart', 'False')
            
            self.Re    = Re
            self.M_inf = M_inf
            self.fn_h5 = fn_h5
            self.nx    = nx
            self.ny    = ny
            
            ## the convection speed per advection cycle (in lattice units)
            self.u_lattice = cs * M_inf
            self.U_inf = self.u_lattice
            
            ## unit rectilinear grid
            self.x = np.arange(nx, dtype=np.float64)
            self.y = np.arange(ny, dtype=np.float64)
            
            self.width  = nx
            self.height = ny
            
            ## set the obstacle
            ## return characteristic length (in lattice units), e.g. height/diameter of obstacle
            self.lchar = self.set_obstacle()
            
            ## the number of timesteps needed for the freestream to advect one lchar
            self.tchar = self.lchar / self.U_inf
            
            ## ν = Δt·cs²·(τ-(1/2))
            self.nu_lattice = self.U_inf * self.lchar / self.Re
            
            ## τ : relaxation timescale / time constant of collision operator
            ## for the BGK method, which describes how quickly the distribution
            ## function relaxes to its equilibrium state.
            self.tau = (3. * self.nu_lattice) + 0.5
            
            self.ts0 = 0
            self.ti_output = 0
        
        if not any([(self.bcNS=='slip_wall'),(self.bcNS=='periodic')]):
            print(">>> bcNS option not valid: '%s'"%(self.bcNS,))
            print(">>> options are 'slip_wall', 'periodic'")
            raise ValueError
        
        even_print('nx', '%i'%nx )
        even_print('ny', '%i'%ny )
        even_print('boundary condition north/south','%s'%self.bcNS)
        print(72*'-')
        even_print('Re'         , '%0.1f [-]' % Re              )
        even_print('M_inf'      , '%0.4f [-]' % M_inf           )
        even_print('u_lattice'  , '%0.8f'     % self.u_lattice  )
        even_print('nu_lattice' , '%0.6e'     % self.nu_lattice )
        even_print('τ'          , '%0.6f'     % self.tau        )
        even_print('lchar'      , '%i'        % self.lchar      )
        even_print('tchar'      , '%0.2f'     % self.tchar      )
        print(72*'-')
        
        ts_per_flowpass = self.tchar ## ts per flowpass (1x lchar/uchar)
        ts_per_frame    = self.data_save_period ## n ts spanned by one frame
        
        mp4_frames_per_flowpass = ts_per_flowpass / ts_per_frame
        mp4_s_per_flowpass      = mp4_frames_per_flowpass / self.fps
        
        '''
        For low Re (e.g. Re≈1000) where the bulk convective mode
        is the dominant unsteadiness, mp4_s_per_flowpass≈0.6 is visually reasonable.
        -----
        For higher Re (e.g. Re≈20000, necessitating higher resolution), there are smaller
        unsteadiness modes which fluctuate at a higher physical frequency.
        So keeping mp4_s_per_flowpass≈0.6 will match the 'visual speed' of the largest hydrodynamic mode, 
        but the finer modes might appear 'too fast'.
        At higher Re, mp4_s_per_flowpass≈2.0 is therefore more visually reasonable.
        '''
        
        even_print('data save period'         , '%i [ts]'%self.data_save_period )
        even_print('mp4 frames per flowpass'  , '%0.2f'%mp4_frames_per_flowpass )
        even_print('mp4 frames per second'    , '%i'%self.fps )
        even_print('mp4 seconds per flowpass' , '%0.2f'%mp4_s_per_flowpass )
        print(72*'-')
        
        ## assert relation relaxation timescale τ : kinematic viscosity ν
        np.testing.assert_allclose(self.nu_lattice, 1*cs**2*(self.tau-0.5), rtol=1e-12)
        
        ## initialize the flow field
        if not self.start_from_restart:
            self.initialize_flowfield()
        
        ## initialize the HDF5 file
        if not self.start_from_restart:
            self.initialize_h5()
            print(72*'-')
    
    def load_restart(self,fn_restart_dat):
        '''
        load a restart and overwrite instance attributes
        '''
        with open(fn_restart_dat,'rb') as f:
            restart_dict = pickle.load(f)
        
        self2 = restart_dict['sim'] ## the serialized lbm class instance
        
        self.fn_h5    = self2.fn_h5
        
        self.obstacle = np.copy( self2.obstacle )
        self.Re       = self2.Re
        self.M_inf    = self2.M_inf
        self.nx       = self2.nx
        self.ny       = self2.ny
        self.width    = self.nx
        self.height   = self.ny
        
        self.u_lattice = self2.u_lattice
        self.U_inf     = self2.U_inf
        self.x         = np.copy( self2.x )
        self.y         = np.copy( self2.y )
        
        self.lchar      = self2.lchar
        self.tchar      = self2.tchar
        self.nu_lattice = self2.nu_lattice
        self.tau        = (3. * self.nu_lattice) + 0.5
        
        self.f_base     = np.copy( self2.f_base )
        self.f          = np.copy( self2.f )
        
        self.data_save_period = self2.data_save_period
        
        self.bcNS = self2.bcNS
        
        ## delete the loaded class instance
        self2 = None; del self2
        
        ## increment ts
        self.ts0       = restart_dict['ts'] + 1
        self.ti_output = restart_dict['ti_output']
        
        return
    
    def write_restart(self, fn_dat_restart, ts=0, ti_output=0):
        '''
        write a restart
        '''
        ## serialize the lbm class instance and save to disk
        print('--w-> %s'%(fn_dat_restart,))
        with open(fn_dat_restart,'wb') as f:
            restart_dict = {'sim':self, 'ts':ts, 'ti_output':ti_output}
            pickle.dump(restart_dict, f, protocol=4)
        return
    
    def set_inlet(self,):
        '''
        apply (quiescent) inlet BC
        - copy from 'baseflow' distribution at x=0 (left/W) boundary
        '''
        i0, iN, iS, iE, iW, iNE, iNW, iSE, iSW = 0,1,2,3,4,5,6,7,8
        for comp_i in [iE,iW,iNE,iNW,iSE,iSW]: ## everything but n0, nN, nS
            self.f[comp_i,0,:] = self.f_base[comp_i,0,:]
        return
    
    def set_slip_walls(self,):
        '''
        apply slip wall BC @ North and South domain boundary
        - mirror distribution components in [y] direction at N and S boundary
        '''
        
        i0, iN, iS, iE, iW, iNE, iNW, iSE, iSW = 0,1,2,3,4,5,6,7,8
        
        f2 = np.copy(self.f)
        
        ## North Boundary
        for i in range(1,self.nx):
            for j in [self.ny-1]:
                self.f[iS ,i,j] = f2[iN ,i,j] ## S  <-- N
                self.f[iSW,i,j] = f2[iNW,i,j] ## SW <-- NW
                self.f[iSE,i,j] = f2[iNE,i,j] ## SE <-- NE
                self.f[iN ,i,j] = f2[iS ,i,j] ## N  <-- S
                self.f[iNW,i,j] = f2[iSW,i,j] ## NW <-- SW
                self.f[iNE,i,j] = f2[iSE,i,j] ## NE <-- SE
        
        ## South Boundary
        for i in range(1,self.nx):
            for j in [0]:
                self.f[iS ,i,j] = f2[iN  ,i,j] ## S  <-- N
                self.f[iSW,i,j] = f2[iNW ,i,j] ## SW <-- NW
                self.f[iSE,i,j] = f2[iNE ,i,j] ## SE <-- NE
                self.f[iN ,i,j] = f2[iS  ,i,j] ## N  <-- S
                self.f[iNW,i,j] = f2[iSW ,i,j] ## NW <-- SW
                self.f[iNE,i,j] = f2[iSE ,i,j] ## NE <-- SE
        
        return
    
    # ===
    
    def set_obstacle(self,):
        '''
        set the obstacle (solid / wall region)
        '''
        
        start_time = timeit.default_timer()
        
        self.obstacle     = np.zeros((self.nx,self.ny),   dtype=bool     )
        self.obstacle_img = np.zeros((self.nx,self.ny,4), dtype=np.uint8 ) ## transparent
        
        if (self.geom_preset=='cylinder') or (self.geom_preset=='circle'):
            
            ## add 0.5 lattice lengths to avoid single pixel point on top __|‾|__
            self.radius   = int(round(0.075*self.height))+0.5
            self.cx       = int(round(0.20*self.width))
            self.cy       = int(round(0.50*self.height))
            self.diameter = 2*self.radius
            lchar         = self.diameter
            
            ## check if inside the circle
            for i in range(self.nx):
                for j in range(self.ny):
                    if ((self.x[i]-self.cx)**2 + (self.y[j]-self.cy)**2 <= self.radius**2):
                        self.obstacle[i,j] = True
                        self.obstacle_img[i,j,:] = [80,80,80,255] ## gray, opaque
        
        elif (self.geom_preset=='square'):
            
            self.side_len = int(round(0.10*self.height))
            self.cx       = int(round(0.25*self.width))
            self.cy       = int(round(0.50*self.height))
            lchar         = self.side_len
            
            ## check if inside the circle
            for i in range(self.nx):
                for j in range(self.ny):
                    if (np.abs(self.x[i]-self.cx)<=(self.side_len/2)) and (np.abs(self.y[j]-self.cy)<=(self.side_len/2)):
                        self.obstacle[i,j] = True
                        self.obstacle_img[i,j,:] = [80,80,80,255] ## gray, opaque
        
        else:
            
            # if (self.fn_bmp is not None) and (self.geom_preset is None): ## read .bmp
            #     raise NotImplementedError
            
            print('geom_preset=%s not recognized!'%(self.geom_preset,))
            raise NotImplementedError
        
        ## save image of obstacle
        self.obstacle_img_green_bg = np.copy(self.obstacle_img)
        for i in range(self.nx):
            for j in range(self.ny):
                if not self.obstacle[i,j]:
                    self.obstacle_img_green_bg[i,j,:] = [0,255,0,255] ## green, opaque
        img_ = np.ascontiguousarray(np.copy( np.flip(self.obstacle_img_green_bg,axis=1).transpose(1,0,2) ))
        mpl.image.imsave('obstacle.bmp', img_ )
        
        even_print('lbm.set_obstacle()','%0.2f [s]'%(timeit.default_timer() - start_time))
        
        return lchar
    
    def initialize_flowfield(self,):
        '''
        set the initial flow field [u,v,ρ]
        '''
        
        start_time = timeit.default_timer()
        
        ## quiescent flow
        ux  = self.u_lattice * np.ones((self.nx,self.ny), dtype=np.float64)
        uy  = 0.             * np.ones((self.nx,self.ny), dtype=np.float64)
        rho =                  np.ones((self.nx,self.ny), dtype=np.float64)
        
        ## 'turbulent' initial flow (N% Gaussian filtered white noise) --> helps initiate unsteadiness
        rng = np.random.default_rng(seed=1)
        sigma = 3
        ux_turb  = np.copy(ux)  + ( 0.02 * self.u_lattice * scipy.ndimage.gaussian_filter( rng.uniform( -1.0,+1.0, size=(self.nx,self.ny)), sigma=sigma, order=0, mode='constant') )
        uy_turb  = np.copy(uy)  + ( 0.02 * self.u_lattice * scipy.ndimage.gaussian_filter( rng.uniform( -1.0,+1.0, size=(self.nx,self.ny)), sigma=sigma, order=0, mode='constant') )
        rho_turb = np.copy(rho) + ( 0.02 *              1 * scipy.ndimage.gaussian_filter( rng.uniform( -1.0,+1.0, size=(self.nx,self.ny)), sigma=sigma, order=0, mode='constant') )
        
        ## removed noised flow inside obstacle
        ux_turb[self.obstacle]  = ux[self.obstacle]
        uy_turb[self.obstacle]  = uy[self.obstacle]
        rho_turb[self.obstacle] = rho[self.obstacle]
        
        self.f_base = equilibrium(rho,      ux,      uy     ) ## base flow:    this gets used at t>0 (e.g. at boundaries)
        self.f      = equilibrium(rho_turb, ux_turb, uy_turb) ## initial flow: this gets used at t=0
        
        even_print('lbm.initialize_flowfield()','%0.2f [s]'%(timeit.default_timer() - start_time))
        
        return
    
    def set_timing(self,):
        '''
        set data write frequency, etc.
        --> this is now simply an __init__ input
        '''
        #self.pic_output_period = 100
        pass
        return
    
    def initialize_h5(self,):
        '''
        initialize the HDF5
        '''
        
        start_time = timeit.default_timer()
        
        #print('--w-> %s'%(self.fn_h5,))
        with h5py.File(self.fn_h5, 'w', libver='latest') as hf:
            
            hf.create_dataset('U_inf'      , data=(self.U_inf,)      , maxshape=(1,))
            hf.create_dataset('Re'         , data=(self.Re,)         , maxshape=(1,))
            hf.create_dataset('nu_lattice' , data=(self.nu_lattice,) , maxshape=(1,))
            hf.create_dataset('lchar'      , data=(self.lchar,)      , maxshape=(1,))
            hf.create_dataset('obstacle'   , data=self.obstacle      , maxshape=np.shape(self.obstacle), dtype=self.obstacle.dtype, chunks=True)
            
            ## rectilinear grid coordinate arrays (1D)
            hf.create_dataset('x', data=self.x)
            hf.create_dataset('y', data=self.y)
            
            ## rectilinear grid coordinate arrays (2D)
            xx, yy = np.meshgrid(self.x, self.y, indexing='ij')
            hf.create_dataset('xx', data=xx.T, chunks=True)
            hf.create_dataset('yy', data=yy.T, chunks=True)
        
        even_print('lbm.initialize_h5()','%0.2f [s]'%(timeit.default_timer() - start_time))
        
        return
    
    def encode_video(self,**kwargs):
        '''
        encode an .mp4 video from the contents of pics/
        '''
        fps = kwargs.get('fps',None)
        if (fps is None):
            fps = self.fps
        
        fn_mp4 = 'umag.mp4'
        adir   = 'pics'
        if shutil.which('ffmpeg') is not None:
            ffmpeg_cmd = ['ffmpeg', '-y', '-framerate', '%i'%(fps,), '-i', adir+'/umag_%06d.png',
                          '-c:v', 'libx264', '-crf', '20', '-preset', 'veryslow', '-profile:v', 'high', 
                          ## '-vf', '"pad=ceil(iw/2)*2:ceil(ih/2)*2"',
                          '-level', '4.0', '-bf', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', fn_mp4]
            ##
            if (shutil.which('wsl.exe') is not None):
                ffmpeg_cmd.insert(0, 'wsl.exe')
            ##
            print(' '.join(ffmpeg_cmd))
            #
            # ffmpeg -y -framerate 60 -i pics/umag_%06d.png -c:v libx264 -crf 20 -preset veryslow -profile:v high -level 4.0 -bf 2 -pix_fmt yuv420p -movflags +faststart umag.mp4
            #
            ##
            subprocess.call(ffmpeg_cmd)
        
        return
    
    def run(self,nts):
        '''
        run simulation: update field in time
        '''
        print('\n'+'lbm.run()')
        print(72*'-')
        even_print('ts0', '%i'%self.ts0)
        even_print('nts', '%i'%nts)
        even_print('ti_output', '%i'%self.ti_output)
        even_print('data save period', '%i [ts]'%self.data_save_period )
        
        precision_bytes = 4 ## save data in single precision (restart is double precision)
        data_b_per_period = 3*precision_bytes*nx*ny
        even_print('data per period', '%0.1f [MB]'%(data_b_per_period/1024**2,))
        
        n_periods_this_run = math.floor(nts/self.data_save_period)
        even_print('n periods', '%i'%(n_periods_this_run,))
        
        n_flowpasses_this_run = math.floor(nts/self.tchar)
        even_print('n flowpasses', '%i'%(n_flowpasses_this_run,))
        
        data_gb_this_run = data_b_per_period * n_periods_this_run / 1024**3
        even_print('data output', '%0.1f [GB]'%(data_gb_this_run,))
        
        if self.start_from_restart:
            ts_vec = np.arange(self.ts0,self.ts0+nts)
        else:
            ts_vec = np.arange(self.ts0,self.ts0+nts+1)
        even_print('first timestep', '%i'%ts_vec[0])
        even_print('last timestep', '%i'%ts_vec[-1])
        
        # === main timestepping loop
        
        print(72*'-')
        progress_bar = tqdm(total=self.ts0+nts, initial=self.ts0, ncols=100, desc='lbm.run()', leave=False, file=sys.stdout)
        
        ts_loop_was_broken = False
        ti_output = self.ti_output
        for ts in ts_vec:
            
            # ============================================================ #
            # physics
            # ============================================================ #
            
            if (ts>0): ## don't advect at ts==0
                
                ## stream (advect) & bounce back
                self.f = stream(self.f,self.obstacle)
                
                ## collision & relaxation
                self.f, rho, ux, uy = collide(self.f,self.tau)
                
                ## set boundary conditions
                if (self.bcNS=='slip_wall'):
                    self.set_slip_walls()
                self.set_inlet()
            
            # ============================================================ #
            # every N timesteps, do:
            # - save data
            # - see if exit/restart write is requested
            # - check for NaNs
            # - plot .pngs
            # ============================================================ #
            
            if (ts%self.data_save_period==0):
                
                rho, ux, uy = get_macros(self.f)
                
                ## write u,v,rho to HDF5
                with h5py.File(self.fn_h5, 'a', libver='latest') as hf:
                    
                    if ('ts_%06d/%s'%(ti_output,'u') in hf):
                        del hf['ts_%06d/%s'%(ti_output,'u')]
                    if ('ts_%06d/%s'%(ti_output,'v') in hf):
                        del hf['ts_%06d/%s'%(ti_output,'v')]
                    if ('ts_%06d/%s'%(ti_output,'rho') in hf):
                        del hf['ts_%06d/%s'%(ti_output,'rho')]
                    
                    hf.create_dataset('ts_%06d/%s'%(ti_output,'u')   , data=ux.astype(np.float32)  , maxshape=np.shape(ux)  , dtype=np.float32, chunks=True)
                    hf.create_dataset('ts_%06d/%s'%(ti_output,'v')   , data=uy.astype(np.float32)  , maxshape=np.shape(uy)  , dtype=np.float32, chunks=True)
                    hf.create_dataset('ts_%06d/%s'%(ti_output,'rho') , data=rho.astype(np.float32) , maxshape=np.shape(rho) , dtype=np.float32, chunks=True)
                
                ## if flow field has NaNs, break timestepping loop
                if np.isnan(ux).any():
                    progress_bar.close()
                    print('NaNs found! exiting!')
                    print(72*'-')
                    break
                
                # === plot
                
                if not os.path.isdir('pics'):
                    Path('pics').mkdir(parents=True, exist_ok=True)
                
                if True: ## plot : umag
                    
                    umag = np.sqrt(ux**2 + uy**2)
                    
                    norm = mpl.colors.Normalize(vmin=0, vmax=2)
                    #cmap = cmocean.cm.balance
                    #cmap = mpl.cm.RdBu_r
                    cmap = cmasher.cm.iceburn
                    
                    fn_png = os.path.join('pics', '%s_%06d.png'%('umag',ti_output))
                    
                    # === plot with imshow()
                    
                    if False:
                        plt.close('all')
                        #plt.style.use('default')
                        fig1 = plt.figure(dpi=200)
                        ax1 = fig1.gca()
                        ##
                        ax1.axis('off')
                        plt.axis('off')
                        ax1.margins(0)
                        ax1.set_aspect('equal')
                        ax1.set_position([0,0,1,1])
                        ##
                        im1          = ax1.imshow(umag.T/self.U_inf, cmap=cmap, norm=norm, origin='lower', interpolation='none', aspect='equal')
                        im1_obstacle = ax1.imshow(self.obstacle_img.transpose(1,0,2),      origin='lower', interpolation='none', aspect='equal')
                        ##
                        dpi_out = self.nx/plt.gcf().get_size_inches()[0]
                        fig1.savefig(fn_png, bbox_inches='tight', pad_inches=0, dpi=dpi_out)
                        #plt.show()
                        pass
                    
                    # === plot with PIL.Image.imwrite()
                    
                    if True:
                        img = np.copy(cmap(norm(umag/self.U_inf), bytes=True))
                        img[self.obstacle] = np.array([120,120,120,255], dtype=np.uint8) ## gray90
                        #img[self.obstacle] = np.array([252,247,94,255], dtype=np.uint8) ## yellow
                        img = img.transpose(1,0,2)
                        img = np.flip(img,axis=0)
                        img = np.copy(img)[:,:,:3]
                        img = PIL.Image.fromarray(img, mode='RGB')
                        img = img.save(fn_png)
                
                if False: ## plot : z-vorticity
                    
                    dudy   = np.gradient(ux, self.y, edge_order=2, axis=1)
                    dvdx   = np.gradient(uy, self.x, edge_order=2, axis=0)
                    vort_z = dvdx - dudy
                    vort_z = np.copy( vort_z )
                    
                    plt.close('all')
                    plt.style.use('default')
                    fig1 = plt.figure(frameon=False)
                    ax1 = fig1.gca()
                    ##
                    ax1.axis('off')
                    plt.axis('off')
                    ax1.margins(0)
                    ax1.set_aspect('equal')
                    ax1.set_position([0, 0, 1, 1])
                    ##
                    ## z-vorticity
                    vort_z_min = 0.0 - 0.07
                    vort_z_max = 0.0 + 0.07
                    #norm1 = mpl.colors.Normalize(vmin=vort_z_min, vmax=vort_z_max)
                    norm = mpl.colors.SymLogNorm(base=10, linthresh=5e-3, vmin=vort_z_min, vmax=vort_z_max)
                    #cmap = cmocean.cm.balance
                    #cmap = mpl.cm.RdBu_r
                    cmap = mpl.cm.PuOr_r
                    #colors_in = [cmap(x) for x in np.linspace(0,1,256)]
                    #cmap = LinearSegmentedColormap.from_list('balance_seg', colors_in, N=64)
                    ##
                    im1          = ax1.imshow(vort_z.T,                           origin='lower', interpolation='none', aspect='equal', cmap=cmap, norm=norm)
                    im1_obstacle = ax1.imshow(self.obstacle_img.transpose(1,0,2), origin='lower', interpolation='none', aspect='equal')
                    ##
                    dpi_out = self.nx/plt.gcf().get_size_inches()[0]
                    fig1.savefig(os.path.join('pics', '%s_%06d.png'%('vort',ti_output)), bbox_inches='tight', pad_inches=0, dpi=dpi_out)
                    #plt.show()
                    pass
                
                # ===
                
                ti_output += 1
                
                # === check for stop request
                
                ## if a file 'stop.txt' is present, save out memory & break timestepping loop
                if os.path.exists('stop.txt'):
                    progress_bar.close()
                    os.remove('stop.txt') ## remove it
                    print('>>> restart exit was requested (stop.txt found)')
                    ts_loop_was_broken = True
                    break
                
                # === check if max wall time exceeded
                
                if (self.max_time_s is not None):
                    time_s = timeit.default_timer() - self.start_time
                    if (time_s >= self.max_time_s):
                        progress_bar.close()
                        print('>>> max wall time (%i [s]) reached!'%(self.max_time_s,))
                        ts_loop_was_broken = True
                        break
            
            progress_bar.update()
        progress_bar.close()
        
        ## write restart
        if not ts_loop_was_broken:
            print('>>> finished %i timesteps'%(nts,))
        print('>>> stopping simulation at ts=%i'%(ts,))
        self.write_restart('restart.dat', ts=ts, ti_output=ti_output)
        
        self.ti_output = ti_output
        self.ts = ts
        
        return

# main
# ======================================================================

if __name__ == '__main__':
    
    ## resolution/Re preset
    #Re, nx, ny, data_save_period, nts = 16000, 3840, 2160, 50, 80000  ## 2160p / 4k
    #Re, nx, ny, data_save_period, nts = 4000,  1920, 1080, 50, 300000 ## 1080p
    #Re, nx, ny, data_save_period, nts = 2000,  1280, 720,  50, 400000 ## 720p
    Re, nx, ny, data_save_period, nts = 1000,  854,  480,  40, 240000 ## 480p
    #Re, nx, ny, data_save_period, nts = 500,   640,  360,  25, 150000 ## 360p
    
    ## check if a restart exists
    fn_restart = Path('restart.dat')
    if fn_restart.exists():
        restart = str(fn_restart)
    else:
        restart = None
    
    ## initialize the simulator, then run N timesteps
    sim = lbm(Re=Re,
              M_inf=0.1,
              fn_h5='data.h5',
              nx=nx,
              ny=ny, 
              geom_preset='cylinder',
              max_time_s=23*3600,
              data_save_period=data_save_period,
              bcNS='slip_wall',
              restart=restart)
    sim.run(nts=nts)
    sim.encode_video()
    
    ## copy data file to backup
    #shutil.copy2('data.h5', 'data_t%i.h5'%(sim.ti_output,))