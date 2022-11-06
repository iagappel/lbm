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

import datetime
import timeit

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cmocean, colorcet, cmasher

import PIL

import mpi4py
from mpi4py import MPI

'''
========================================================================

Plot script for LBM simulation data file (HDF5)
- mpi parallel, run with:

$> mpiexec -n 8 python3 lbm_plot.py

========================================================================
'''

# main
# ======================================================================

if __name__ == '__main__':
    
    comm    = MPI.COMM_WORLD
    rank    = comm.Get_rank()
    n_ranks = comm.Get_size()
    
    if rank==0:
        verbose = True
    else:
        verbose = False
    
    fn_h5 = 'data.h5'
    
    with h5py.File(fn_h5,'r',driver='mpio',comm=comm) as f1:
        
        obstacle = np.copy( f1['obstacle'][()] )
        U_inf    = f1['U_inf'][0]
        x        = np.copy( f1['x'][()] )
        y        = np.copy( f1['y'][()] )
        
        ## get ts_XXXXXX datasets
        dset_names = list(f1.keys())
        tss = sorted([ dss for dss in dset_names if re.match(r'ts_\d',dss) ])
        ti = np.array( sorted([ int(re.findall(r'[0-9]+', tss_)[0]) for tss_ in tss ]), dtype=np.int32 )
        
        if not np.all( np.diff(ti)==1 ):
            raise AssertionError('ti not constant')
        
        nt = ti.shape[0]
        
        ## number of frames to plot on this rank
        rt = n_ranks
        rtl_ = np.array_split(np.array(range(nt),dtype=np.int64),min(rt,nt))
        rtl = [[b[0],b[-1]+1] for b in rtl_ ]
        rt1,rt2 = rtl[rank]
        ntr = rt2 - rt1
        
        if not os.path.isdir('pics'):
            if (rank==0):
                Path('pics').mkdir(parents=True, exist_ok=True)
        
        # ===
        
        if verbose: progress_bar = tqdm(total=ntr, ncols=100, desc='anim', leave=False, file=sys.stdout)
        for tii in range(rt1,rt2):
            
            u = np.copy( f1['ts_%06d/u'%(ti[tii],)][()] )
            v = np.copy( f1['ts_%06d/v'%(ti[tii],)][()] )
            
            # === umag
            
            if True:
                
                umag = np.sqrt(u**2 + v**2)
                
                norm = mpl.colors.Normalize(vmin=0, vmax=2)
                cmap = cmasher.cm.iceburn
                
                fn_png = os.path.join('pics', '%s_%06d.png'%('umag',ti[tii]))
                
                if True:
                    img = np.copy(cmap(norm(umag/U_inf), bytes=True))
                    img[obstacle] = np.array([120,120,120,255], dtype=np.uint8) ## gray90
                    img = img.transpose(1,0,2)
                    img = np.flip(img,axis=0)
                    img = np.copy(img)[:,:,:3]
                    img = PIL.Image.fromarray(img, mode='RGB')
                    img = img.save(fn_png)
            
            # === z-vorticity
            
            if False:
                
                dudy   = np.gradient(u, y, edge_order=2, axis=1)
                dvdx   = np.gradient(v, x, edge_order=2, axis=0)
                vort_z = dvdx - dudy
                
                # sign = np.copy( np.sign(vort_z) )
                # #vort_z = np.log10( np.abs( vort_z) )
                # vort_z = np.log( 1 + np.abs( vort_z) )
                # vort_z *= sign
                
                vort_z_min = 0.0 - 3e-2
                vort_z_max = 0.0 + 3e-2
                #norm = mpl.colors.Normalize(vmin=vort_z_min, vmax=vort_z_max)
                norm = mpl.colors.SymLogNorm(base=10, linthresh=5e-4, vmin=vort_z_min, vmax=vort_z_max) 
                #cmap = cmasher.cm.iceburn
                cmap = cmasher.cm.redshift
                
                fn_png = os.path.join('pics', '%s_%06d.png'%('vort',ti[tii]))
                
                if True:
                    img = np.copy(cmap(norm(vort_z), bytes=True))
                    img[obstacle] = np.array([120,120,120,255], dtype=np.uint8) ## gray90
                    img = img.transpose(1,0,2)
                    img = np.flip(img,axis=0)
                    img = np.copy(img)[:,:,:3]
                    img = PIL.Image.fromarray(img, mode='RGB')
                    img = img.save(fn_png)
            
            # === angle
            
            if False:
                
                angle = np.arctan2(v,u)
                norm = mpl.colors.Normalize(vmin=-np.pi, vmax=+np.pi)
                #cmap = cmasher.cm.infinity
                cmap = mpl.cm.twilight
                
                fn_png = os.path.join('pics', '%s_%06d.png'%('angle',ti[tii]))
                
                if True:
                    img = np.copy(cmap(norm(angle), bytes=True))
                    img[obstacle] = np.array([120,120,120,255], dtype=np.uint8) ## gray90
                    img = img.transpose(1,0,2)
                    img = np.flip(img,axis=0)
                    img = np.copy(img)[:,:,:3]
                    img = PIL.Image.fromarray(img, mode='RGB')
                    img = img.save(fn_png)
            
            if verbose: progress_bar.update()
        if verbose: progress_bar.close()
    
    # ===
    
    comm.Barrier()
    
    adir = 'pics'
    
    if True:
        
        fn_mp4 = 'vort.mp4'
        if (rank==0):
            if shutil.which('ffmpeg') is not None:
                ffmpeg_cmd = ['ffmpeg', '-y', '-framerate', '60', '-i', adir+'/vort_%06d.png',
                              '-c:v', 'libx264', '-crf', '20', '-preset', 'veryslow', '-profile:v', 'high', 
                              ## '-vf', '"pad=ceil(iw/2)*2:ceil(ih/2)*2"',
                              '-level', '4.0', '-bf', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', fn_mp4]
                ##
                if shutil.which('wsl.exe') is not None:
                    ffmpeg_cmd.insert(0, 'wsl.exe')
                ##
                subprocess.call(ffmpeg_cmd)