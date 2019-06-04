
"""
Importage
"""


# Generic stuff

from copy import copy,deepcopy
import os,sys,glob
from itertools import product
from datetime import datetime
from numpy import exp,sin,cos,sqrt,pi, r_,floor,zeros
import numpy as np,pandas as pd
from joblib import Parallel,delayed
from numpy.random import rand,normal
import scipy


# Neuroimaging & signal processing stuff

from scipy.signal import welch,periodogram
from mne.connectivity import spectral_connectivity
from mne.io import RawArray
from mne import create_info
from mne.time_frequency import psd_welch
from mne.surface import decimate_surface
from scipy.spatial import cKDTree
from nilearn.plotting import plot_surf,plot_surf_stat_map


# Vizualization stuff

from matplotlib.tri import Triangulation
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import seaborn as sns
import moviepy
from moviepy.editor import ImageSequenceClip
from matplotlib.pyplot import subplot



"""
Run sim function
"""

# Notes:
# - time series not returned by default (to avoid memory errors). Will be returned 
#   if return_ts=True
# - freq vs. amp param sweep has default range within which max freqs are calculated 
#   of 5-95 Hz

def run_net_sim(wee = 0.5,wei = 1.,wie = -2.,wii = -0.5,wertn = 0.6,weth = .6,
            wthi = 0.2,wthe = 1.65,wrtnth = -2.,wthrtn = 2.,D_e = .0001,D_i= .0001,
            D_th = 0.0001,D_rtn = 0.0001,T = 1024*4,P=1,Q = 1,K = 1,Dt = 0.001,
            dt = 0.1,gain = 20.,threshold = 0.,Pi = 3.14159,g = -0.9,a_e = 0.3,
            a_i = 0.5,a_th = 0.2,a_rtn = 0.2,i_e = -0.35,i_i = -0.3,i_th = 0.5,
            i_rtn = -0.8,tau1  = 20.,tau2 = 5.,I_o = 0.,T_transient=1000,
            stim_freq=35.,stim_amp=0.5,return_ts=False,compute_connectivity=False,
            weights=None,delays=None,
            fmin = 1.,fmax = 100.,fskip = 1,
            stim_nodes='all',stim_pops=['e'],mne_welch=False,#filt_freqs = [0,np.inf],
            stim_type='sinewave',stim_on=None,stim_off=None,
               welch_nperseg=None,welch_noverlap=None):
    
    """
    Time units explained:


    Elements t of Ts are integers and represent 'time units'
  
    Physical duration of a time unit is as follows:
    
    Time (real, in ms) = t*dt*scaling = t*Dt 
    
    scaling is implicitly defined as Dt/dt
    
    Example:  (to add)
   
    
    NOTE: need to be careful with delays units etc. 
    
    """
    
    
    
    
    
    n_nodes = K
    
    x1 = n_nodes/2.
    x2 = x1+20.

    # if I_o is a single number, make it a uniform vector
    if type(I_o) ==float:  I_o = np.ones(n_nodes)*I_o
    
    # Neuronal response function
    def f(u,gain=gain,threshold=threshold):
        output= 1./(1.+exp(-gain*(u-threshold))); 
        return output 
    
    # Initialize variables
    Xi_e = np.random.uniform(low=-1.,high=1.,size=[K,T+T_transient])
    Xi_i = np.random.uniform(low=-1.,high=1.,size=[K,T+T_transient])
    Xi_th = np.random.uniform(low=-1.,high=1.,size=[K,T+T_transient])
    Xi_rtn = np.random.uniform(low=-1.,high=1.,size=[K,T+T_transient])
    
    e = np.zeros_like(Xi_e) 
    i = np.zeros_like(Xi_i)
    th = np.zeros_like(Xi_th)
    rtn = np.zeros_like(Xi_rtn)
    
    Ts = np.arange(0,T+T_transient)  # time step numbers list
    

    if stim_type == 'sinewave':
        #state_input = ((Ts>x1+1)&(Ts<x2)).astype(float)
        state_input = 1. # not sure what this is suppose do be doing...
        stim = state_input * stim_amp*sin(2*Pi*stim_freq*Ts*Dt)
    elif stim_type == 'spiketrain':
        stim_period =1./(stim_freq*Dt)
        spiketrain = np.zeros_like(Ts)
        t_cnt = 0
        t = 0
        while t < Ts[-4]:
            t+=1
            if t>=t_cnt:
                spiketrain[t] = stim_amp
                spiketrain[t+1] = stim_amp
                spiketrain[t+2] = 0
                spiketrain[t+3] = 0
                t_cnt = t_cnt+stim_period
        stim = spiketrain
    elif stim_type== 'square': 
        stim = np.zeros_like(Ts)
        stim[(Ts>=stim_on) & (Ts<=stim_off)] = stim_amp
    elif stim_type== 'rand':
        stim = np.random.randn(Ts.shape[0])*stim_amp
        
        

    # convert stim to array the size of e
    stim = np.tile(stim,[n_nodes,1])

    # arrange stimulus for the nodes
    if stim_nodes == 'all': stim_nodes = np.arange(0,n_nodes)
    nostim_nodes = np.array([k for k in range(n_nodes) if k not in stim_nodes])
    if nostim_nodes.shape[0] != 0: stim[nostim_nodes,:] = 0.

    # decide which population to add the stimulus (currently mutually exclusive)
    stim_e = stim_i = stim_th = stim_rtn = np.zeros_like(stim)
    if 'e' in stim_pops: stim_e = stim
    if 'i' in stim_pops: stim_i = stim
    if 'th' in stim_pops: stim_th = stim
    if 'rtn' in stim_pops: stim_rtn = stim
            
            
            
        

    # If I do this, it changes the time series dramatically (halves the period...)
    #e = Xi_e
    #i = Xi_i
    #th = Xi_th
    #rtn = Xi_rtn
    # ...so initialize just first state with random variables instead...
    e[:,0] = np.random.randn(n_nodes)
    i[:,0] = np.random.randn(n_nodes)
    th[:,0] = np.random.randn(n_nodes)
    rtn[:,0] = np.random.randn(n_nodes)
    
    # Define time
    #ts = r_[0:sim_len:dt] # integration step time points (ms)
    #n_steps = len(ts)  #n_steps = int(round(sim_len/dt)) # total number of integration steps
    #ts_idx = arange(0,n_steps) # time step indices
    
    # Sampling frequency for fourier analysis
    fs = (1000.*Dt)/dt
    #fs = 1./dt
    

    # Make a past value index lookup table from 
    # the conduction delays matrix and the 
    # integration step size
    #delays_lookup = floor(delays/dt).astype(int)  
    delays_lookup = floor(delays).astype(int)  
   
    maxdelay = int(np.max(delays_lookup))
    
    # Integration loop
    for t in Ts[maxdelay:-1]:
    
    
        # For each node, find and summate the time-delayed, coupling 
        # function-modified inputs from all other nodes
        ccin = zeros(n_nodes) 
    
        for j in range(0,n_nodes):
        
            # The .sum() here sums inputs for the current node here 
            # over both nodes and history (i.e. over space-time)
            #insum[i] = f((1./n_nodes) * history_idx_weighted[:,i,:] * history).sum() 

            for k in range(0,n_nodes):
                
                # Get time-delayed, sigmoid-transformed inputs
                delay_idx = delays_lookup[j,k] 
                if delay_idx < t: 
                    delayed_input = f(e[k,t-delay_idx])
                else:
                    delayed_input = 0
                    
                # Weight  inputs by anatomical connectivity, and normalize by number of nodes
                ccin[j]+=(1./n_nodes)*weights[j,k]*delayed_input
            
            
        # Cortico-cortical connections
        weefe = wee*f(e[:,t])
        wiifi = wii*f(i[:,t])
        wiefi = wie*f(i[:,t])
        weife = wei*f(e[:,t]) 
        gccin = g*ccin
            
        # Cortico-thalamic connections
        wethfe = weth*f(e[:,int(t-tau1)]) # t-tau1]) 
        wertnfe = wertn*f(e[:,int(t-tau1)]) #t-tau1])
            
        # Thalamo-cortical connections
        if t<=tau1:
            wthefth=wthifth=0
        else:
            wthefth = wthe*f(th[:,int(t-tau1)]) #t-tau1])
            wthifth = wthi*f(th[:,int(t-tau1)]) # t-tau1])
            
        # Thalamo-thalamic connections
        if t<=tau2:
            wthrtnfth=wrtnthfrtn=0
        else:
            wrtnthfrtn = wrtnth*f(rtn[:,int(t-tau2)]) # t-tau2])
            wthrtnfth = wthrtn*f(th[:,int(t-tau2)]) # t-tau2])

        # Noise inputs
        e_noise = sqrt(2.*D_e*dt)*Xi_e[:,t]
        i_noise = sqrt(2.*D_i*dt)*Xi_i[:,t]
        th_noise = sqrt(2.*D_th*dt)*Xi_th[:,t]
        rtn_noise = sqrt(2.*D_rtn*dt)*Xi_rtn[:,t]
            
            
        # Collate inputs
        e_inputs = weefe + wiefi + wthefth + gccin + i_e + stim_e[:,t]
        i_inputs = weife + wiifi + wthifth + i_i         + stim_i[:,t]
        th_inputs = wethfe + wrtnthfrtn + I_o + i_th     + stim_th[:,t]
        rtn_inputs = wthrtnfth + wertnfe+ i_rtn          + stim_rtn[:,t]
          
        # d/dt
        e[:,t+1]   = e[:,t]   + dt*a_e*(-e[:,t]     + e_inputs)   + e_noise
        i[:,t+1]   = i[:,t]   + dt*a_i*(-i[:,t]     + i_inputs)   + i_noise
        th[:,t+1]  = th[:,t]  + dt*a_th*(-th[:,t]   + th_inputs)  + th_noise
        rtn[:,t+1] = rtn[:,t] + dt*a_rtn*(-rtn[:,t] + rtn_inputs) + rtn_noise

        #x[t_it+1,:] = x[t_it,:] + dt * alpha * (-x[t_it,:] + g * rec_input) + noise + 
        #S[t_it,:]

    # Concatenate sim time series
    df_all = pd.concat({'e': pd.DataFrame(e.T[T_transient:]),
                        'i': pd.DataFrame(i.T[T_transient:]),
                        'th':pd.DataFrame(th.T[T_transient:]),
                        'rtn': pd.DataFrame(rtn.T[T_transient:]),
                        'stim': pd.DataFrame(stim.T[T_transient:])},axis=1)
    df_all.index = (Ts[T_transient:]-T_transient)*(Dt*1000.)

    # Compute power spectra:

    
    if mne_welch == False:
    
        # 1. Welch's method
        tmp = welch(df_all.T,fs= 1./Dt,nperseg=welch_nperseg,noverlap=welch_noverlap)
        df_wel = pd.DataFrame(tmp[1].T,index=tmp[0])
        df_wel.columns = df_all.columns
    

    else: 
        
        # MNE Plot
        dat = df_all #     sim9a1[0]['e']
        nchan = dat.shape[1]# meg.shape[0]
        ch_names  = ['%s_%s' %(a,b) for a,b in np.array(dat.columns)]
        #ch_names = list(dat.columns.values.astype(str))
        ch_types = ['mag' for _ in range(nchan)]
        raw = RawArray(dat.T/1e9, 
               create_info(ch_names,1e3/(dat.index[1] - dat.index[0]),ch_types),verbose=False)
        
        tmin = 0; tmax = raw.times[-1]
        #fmin,fmax = filt_freqs
        
        raw.filter(fmin,fmax)
        
        n_fft = min(np.diff(raw.time_as_index([tmin, tmax]))[0] + 1, 2048)
        # probably will return 2048
        psds,freqs = psd_welch(raw,n_fft=n_fft,fmin=fmin,fmax=fmax,verbose=False)
        df_wel = pd.DataFrame(psds.T,index=freqs)#blah[0].T,index=blah[1])
        df_wel.columns = df_all.columns

    
    # 2. Periodogram
    tmp = periodogram(df_all.T,fs= 1./Dt) 
    df_per = pd.DataFrame(tmp[1].T,index=tmp[0])
    df_per.columns = df_all.columns
    
    
    if compute_connectivity:
        
        cols = df_all['e'].columns
        data = df_all['e'].values.copy().T[np.newaxis,:,:]

        
        spec_conn_coh = spectral_connectivity(data=data,method='coh',sfreq=1./Dt,verbose=False,
                                              fmin=fmin,fmax=fmax,fskip=fskip)
        con_coh,freqs_coh,times_coh,n_epochs_coh,n_tapers_coh = spec_conn_coh

        df_con_coh = pd.concat({f: pd.DataFrame(((con_coh[:,:,f_it] + con_coh[:,:,f_it].T)/2.).T,
                            columns= cols,index=cols) for f_it,f in enumerate(freqs_coh)})
        df_con_coh.index.names = ['freq', 'region']        
        

        
        spec_conn_plv = spectral_connectivity(data=data,method='plv',sfreq=1./Dt,verbose=False,
                                              fmin=fmin,fmax=fmax,fskip=fskip)
        con_plv,freqs_plv,times_plv,n_epochs_plv,n_tapers_plv = spec_conn_plv

        df_con_plv = pd.concat({f: pd.DataFrame(((con_plv[:,:,f_it] + con_plv[:,:,f_it].T)/2.).T,
                            columns= cols,index=cols) for f_it,f in enumerate(freqs_plv)})
        df_con_plv.index.names = ['freq', 'region']        
        
        
        df_con_corr = df_all['e'].corr()
        
    else:
        df_con_coh=df_con_plv=df_con_corr=None
        
    # this is necessary to stop memory error crashes
    if return_ts==False: df_all=None
    
    return df_all,df_wel,df_per,df_con_coh,df_con_plv,df_con_corr











def sim_summary(simres,fname=None,singlenode=False,fig_title='',text_info='',mne_welch=True,
                fmin=0,fmax=100.):

    fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(12,12))

    
    #fig.suptitle(fig_title)
    
    
    a = ax[0][0]
    a.axis('off')
    a.grid('off')
    a.set_title('\n\n' + fig_title, fontsize=20)
    a.text(0,0.5,text_info)
    
    
    a = ax[0][1]
    df = simres[0]['e'].loc[:1000] # 1000:2000]
    df.plot(legend=False,c='k', linewidth=0.5,alpha=0.5,ax=a)
    a.set_title('time series')


    a = ax[1][1]
    df = simres[2]['e'].loc[:80]
    df.plot(legend=False,c='k', linewidth=0.5,ax=a)#,alpha=0.5)#,ax=a)
    a.set_title('periodogram')

    

    a = ax[1][0]
    if not mne_welch:
        
        df = simres[1]['e'].loc[:80]
        df.plot(legend=False,c='k', linewidth=0.5,ax=a)#,alpha=0.5)#,ax=a)
        a.set_title('welch')

    else:
            
        dat = simres[0]['e']
        nchan = dat.shape[1]# meg.shape[0]
        ch_names = list(dat.columns.values.astype(str))
        ch_types = ['mag' for _ in range(nchan)]
        raw = RawArray(dat.T/1e9, create_info(ch_names,1e3/(dat.index[1] - dat.index[0]),ch_types),verbose=False)
        raw.filter(fmin, fmax)
        # raw.plot_psd(fmin=fmin,fmax=fmax,average=False,ax=a);

        tmin = 0; tmax = raw.times[-1]
        n_fft = min(np.diff(raw.time_as_index([tmin, tmax]))[0] + 1, 2048) # probably will return 2048
        psds,freqs = psd_welch(raw,n_fft=n_fft,fmin=fmin,fmax=fmax,verbose=False)
        df_psds = pd.DataFrame(psds.T,index=freqs)#blah[0].T,index=blah[1])
        df_psds.loc[:fmax].plot(legend=False,logy=True,c='k', alpha=0.05,linewidth=0.5,ax=a)
        a.set_title('welch (mne)')
        
    
    
    a = ax[2][0]
    df = simres[2]['e']#.loc[:80]
    df.plot(logx=True,logy=True,ax=a,legend=False,c='k',alpha=0.5)

    pl_line = 1./df.index.values
    pl_line[pl_line=='inf'] = 0
    pl2_line = 1./(df.index.values**2)
    pl2_line[pl2_line=='inf'] = 0
    newdf = pd.DataFrame([pl_line,pl2_line],index=['1/f', '1/f^2'], columns=df.index).T
    newdf.plot(logx=True,logy=True,ax=a)
    a.set_title('Log power')
    #df.plot(logx=True,logy=True,ax=a)
    a.set_ylim([10E-12,10E0])


    a = ax[2][1]
    if singlenode==True:
        a.axis('off'); a.grid('off')
    else:
        sns.heatmap(simres[0]['e'].corr(),ax=a,cmap='RdBu_r',vmin=-1,vmax=1,
                    mask=np.eye(simres[0]['e'].corr().shape[0]))# ,vmax=0.3)
        a.set_title('pearson correlations')
                
    a = ax[3][0]
    if singlenode==True:
        a.axis('off'); a.grid('off')
    else:
        sns.heatmap(simres[1]['e'].loc[:100].iloc[::-1],ax=a)
        a.set_title('power over nodes')
        a.set_xlabel('node #')
        a.set_ylabel('freq')
        
    a = ax[3][1]
    if singlenode==True:
        a.axis('off'); a.grid('off')
    else:
        simres[1].idxmax().unstack(0)['e'].plot(kind='bar',ax=a)
        a.set_title('dominant freqs')
        a.set_xlabel('node #')
        a.set_xticklabels('')
        a.set_ylabel('freq')

    plt.tight_layout()

    
    if fname:
        plt.savefig(fname, bbox_inches='tight', transparent=True)
        plt.close()
    
    

"""
Parallellization wrappers
"""


def pse_par(paramslist,sim_func=None,n_jobs=2,temp_folder = '/tmp'):
    
    # Run parallel sims
    sim_res = Parallel(n_jobs=n_jobs,temp_folder=temp_folder)\
             (delayed(sim_func)(**p)\
             for p in paramslist)

    return sim_res        



def FreqVsAmpPsweep(fixed_params,freqs,amps,n_jobs=2,verbose=True,
                    temp_folder='/tmp',outfile=None,outpfx=None,
                    concat_res=False,return_ts=False,peak_freq_range=[5,95],
                    compute_connectivity=False,sim_type='vec',
                    weights=None,delays=None):
    
    sim_func = run_net_sim
    pfr = peak_freq_range
    
    if verbose == True:
        start = datetime.now()
        print 'started: %s' %start
        
    # Set up params list
    combs = list(product(freqs,amps))
    paramslist = []
    for f,a in combs:
        newparams = copy(fixed_params)
        newparams['stim_freq'] = f
        newparams['stim_amp'] = a
        newparams['return_ts'] = return_ts
        newparams['compute_connectivity'] = compute_connectivity
        if weights!=None: newparams['weights'] = weights
        if delays!=None: newparams['delays'] = delays
            
        paramslist.append(newparams)

    # Run parallel sims
    sim_res = pse_par(paramslist,sim_func = sim_func,n_jobs=n_jobs,
                      temp_folder=temp_folder)
    
    # Assemble outputs
    
    _all_dat,_all_wel,_all_per,_all_coh,_all_plv,_all_corr= {},{},{},{},{},{}
    for (f,a),(dat,wel,per,coh,plv,corr) in zip(combs,sim_res):
        _all_dat[f,a] = dat
        _all_wel[f,a] = wel
        _all_per[f,a] = per
        _all_coh[f,a] = coh
        _all_plv[f,a] = plv
        _all_corr[f,a] = corr

    # added in this option because getting memory error if the param space is too big 
    if concat_res == False:
        res_dict = dict(_all_dat=_all_dat,_all_wel=_all_wel,_all_per=_all_per,
                        _all_coh=_all_coh,_all_plv=_all_plv,_all_corr=_all_corr,paramslist=paramslist)
    else:
        
        if return_ts == True:
            all_dat = pd.concat(_all_dat)
            all_dat.index.names = ['stim_freq', 'stim_amp', 't']
            all_dat.columns.names = ['pop', 'region']
        else:
            all_dat = None
            
        all_wel = pd.concat(_all_wel)
        all_wel.index.names = ['stim_freq', 'stim_amp','Hz']
        all_wel.columns.names = ['pop', 'region']

        all_per = pd.concat(_all_per)
        all_per.index.names = ['stim_freq','stim_amp', 'Hz']
        all_per.columns.names = ['pop', 'region']
        
        all_wel_maxfreqs = all_wel.unstack(['stim_freq','stim_amp']).loc[pfr[0]:pfr[1]]\
                                  .idxmax().unstack('region').mean(axis=1).unstack('pop')
        
        all_wel_maxamps = all_wel.unstack(['stim_freq','stim_amp']).loc[pfr[0]:pfr[1]]\
                                  .max().unstack('region').mean(axis=1).unstack('pop')
    
        all_per_maxfreqs = all_per.unstack(['stim_freq','stim_amp']).loc[pfr[0]:pfr[1]]\
                               .idxmax().unstack('region').mean(axis=1).unstack('pop')
        
        all_per_maxamps = all_per.unstack(['stim_freq','stim_amp']).loc[pfr[0]:pfr[1]]\
                               .max().unstack('region').mean(axis=1).unstack('pop')

        res_dict = dict(all_dat=all_dat,all_per=all_per,all_wel=all_wel,
                        all_wel_maxfreqs=all_wel_maxfreqs,
                        all_wel_maxamps=all_wel_maxamps,
                        all_per_maxfreqs=all_per_maxfreqs,
                        all_per_maxamps=all_per_maxamps)

    
    if verbose == True:
        
        finish = datetime.now()
        dur = str(finish - start)
        print 'finished: %s' %finish
        print 'duration: %s' %dur
        
        
    
    return res_dict




def CtxgVsCtxvelPsweep(fixed_params,gs,vels,n_jobs=2,verbose=True,
                        temp_folder='/tmp',outfile=None,outpfx=None,
                        concat_res=False,return_ts=False,peak_freq_range=[5,95],
                        compute_connectivity=False,sim_type='vec',
                        weights=None,delays=None,stim_amp=0.,stim_freq=35.,
                        return_orig=True,concat_ts=False,concat_psds=True):
    
    sim_func = run_net_sim
    
    pfr = peak_freq_range
    
    if verbose == True:
        start = datetime.now()
        print 'started: %s' %start
        
    # Set up params list
    combs = list(product(gs,vels))
    paramslist = []
    for g,vel in combs:
        
        conn=  connectivity.Connectivity(load_default=True)
        conn.speed = vel
        conn.configure()
        n_nodes = conn.number_of_regions
        weights = conn.weights
        delays = conn.delays

        newparams = copy(fixed_params)
        newparams['weights'] = weights
        newparams['delays'] = delays
        newparams['K'] = n_nodes
        newparams['g'] = g
        newparams['stim_amp'] = stim_amp
        newparams['stim_freq'] = stim_freq
        newparams['return_ts'] = return_ts
        newparams['compute_connectivity'] = compute_connectivity
        #if weights!=None: newparams['weights'] = weights
        #if delays!=None: newparams['delays'] = delays
            
        paramslist.append(newparams)

    # Run parallel sims
    sim_res = pse_par(paramslist,sim_func = sim_func,n_jobs=n_jobs,
                      temp_folder=temp_folder)
    
    # Assemble outputs
    _all_dat,_all_per,_all_wel,_all_coh,_all_plv,_all_corr = {},{},{},{},{},{}
    for (g,vel),(dat,per,wel,coh,plv,corr) in zip(combs,sim_res):
        _all_dat[g,vel] = dat
        _all_per[g,vel] = per
        _all_wel[g,vel] = wel
        _all_coh[g,vel] = coh
        _all_plv[g,vel] = plv
        _all_corr[g,vel] = corr

        
    
    if return_orig == True:
        res_dict = dict(_all_dat=_all_dat,_all_per=_all_per,_all_wel=_all_wel,
                        _all_coh=_all_coh,_all_plv=_all_plv,_all_corr=_all_corr,
                         paramslist=paramslist)
    else:
        res_dict = {}

    if concat_ts == True:
            all_dat = pd.concat(_all_dat)
            all_dat.index.names = ['ctx_gain', 'ctx_vel', 't']
            all_dat.columns.names = ['pop', 'region']
            res_dict['all_dat'] = all_dat
            
    if concat_psds==True:
            
        all_wel = pd.concat(_all_wel)
        all_wel.index.names = ['ctx_gain', 'ctx_vel','Hz']
        all_wel.columns.names = ['pop', 'region']

        all_per = pd.concat(_all_per)
        all_per.index.names = ['ctx_gain', 'ctx_vel', 'Hz']
        all_per.columns.names = ['pop', 'region']
         
        all_wel_maxfreqs = all_wel.unstack(['ctx_gain', 'ctx_vel']).loc[pfr[0]:pfr[1]]\
                                  .idxmax().unstack('region').mean(axis=1).unstack('pop')
        
        all_wel_maxamps = all_wel.unstack(['ctx_gain', 'ctx_vel']).loc[pfr[0]:pfr[1]]\
                                 .max().unstack('region').mean(axis=1).unstack('pop')
    
        all_per_maxfreqs = all_per.unstack(['ctx_gain', 'ctx_vel']).loc[pfr[0]:pfr[1]]\
                               .idxmax().unstack('region').mean(axis=1).unstack('pop')
        
        all_per_maxamps = all_per.unstack(['ctx_gain', 'ctx_vel']).loc[pfr[0]:pfr[1]]\
                              .max().unstack('region').mean(axis=1).unstack('pop')


        all_wel_maxfreqs_an = all_wel.unstack(['ctx_gain','ctx_vel']).loc[pfr[0]:pfr[1]]\
                                     .idxmax().unstack(['pop', 'region'])
            
        all_wel_maxamps_an = all_wel.unstack(['ctx_gain','ctx_vel']).loc[pfr[0]:pfr[1]]\
                                     .max().unstack(['pop', 'region'])
            
        all_per_maxfreqs_an = all_per.unstack(['ctx_gain','ctx_vel']).loc[pfr[0]:pfr[1]]\
                                     .idxmax().unstack(['pop', 'region'])
            
        all_per_maxamps_an = all_per.unstack(['ctx_gain','ctx_vel']).loc[pfr[0]:pfr[1]]\
                                     .max().unstack(['pop', 'region'])
            
            
        res_dict['all_per'] = all_per
        res_dict['all_wel'] = all_wel
        res_dict['all_wel_maxfreqs'] = all_wel_maxfreqs
        res_dict['all_wel_maxamps'] = all_wel_maxamps
        res_dict['all_per_maxfreqs'] = all_per_maxfreqs
        res_dict['all_per_maxamps'] = all_per_maxamps
        
        res_dict['all_per_maxfreqs_an'] = all_per_maxfreqs_an
        res_dict['all_per_maxamps_an'] = all_per_maxamps_an
        res_dict['all_wel_maxfreqs_an'] = all_wel_maxfreqs_an
        res_dict['all_wel_maxamps_an'] = all_wel_maxamps_an
        
    
    if verbose == True:
        
        finish = datetime.now()
        dur = str(finish - start)
        print 'finished: %s' %finish
        print 'duration: %s' %dur
        
        
    
    return res_dict







"""
Plotting functions
"""

def plot_heatmaps(dfs,tits,outfile=None,cmap='jet',vmins=[0],vmaxs=[0.1],
                  x_rot=45,y_rot=0,xlabs=None,xylabs=None):
    
    
    if len(dfs) == 4:
        fig, ax = plt.subplots(ncols=2,nrows=2,figsize=(12,6))
        
    for df_it,df in enumerate(dfs):
        thisdf = df.copy()
        #idx = np.round(thisdf.index.copy())
        #cols = np.round(thisdf.columns.copy())
        a = ax.ravel()[df_it]

        idx = ['%1.2f' %i for i in thisdf.index]
        cols = ['%1.2f' %i for i in thisdf.columns]
        
        thisdf.index = idx 
        thisdf.columns = cols

        if len(vmins) == 1:
            thisvmin = vmins[0]
        else:
            thisvmin = vmins[df_it]
            
        if len(vmaxs) == 1:
            thisvmax = vmaxs[0]
        else:
            thisvmax = vmaxs[df_it]
            
        sns.heatmap(thisdf,ax=a,vmin=thisvmin,vmax=thisvmax,cmap=cmap)        
        
        a.set_title(tits[df_it])
    
        if xylabs!=None:
            a.set_xlabel(xylabs[df_it][0])
            a.set_ylabel(xylabs[df_it][1])
    
    
    plt.tight_layout()
    
    
    for a in ax.ravel(): 
        a.set_xticklabels(a.get_xticklabels(),rotation=x_rot,minor=False);
        a.set_yticklabels(a.get_yticklabels(),rotation=y_rot,minor=False);

    if outfile:
        plt.savefig(outfile, bbox_inches='tight', transparent=True)
        
        
#figf = 'cns_poster_figs/sim2b_ts_and_ps.png'


def sim_summary(simres,fname=None,singlenode=False,fig_title='',text_info='',mne_welch=True,
                fmin=0,fmax=100.):

    fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(12,12))

    
    #fig.suptitle(fig_title)
    
    
    a = ax[0][0]
    a.axis('off')
    a.grid('off')
    a.set_title('\n\n' + fig_title, fontsize=20)
    a.text(0,0.5,text_info)
    
    
    a = ax[0][1]
    df = simres[0]['e'].loc[:1000] # 1000:2000]
    df.plot(legend=False,c='k', linewidth=0.5,alpha=0.5,ax=a)
    a.set_title('time series')


    a = ax[1][1]
    df = simres[2]['e'].loc[:80]
    df.plot(legend=False,c='k', linewidth=0.5,ax=a)#,alpha=0.5)#,ax=a)
    a.set_title('periodogram')

    

    a = ax[1][0]
    if not mne_welch:
        
        df = simres[1]['e'].loc[:80]
        df.plot(legend=False,c='k', linewidth=0.5,ax=a)#,alpha=0.5)#,ax=a)
        a.set_title('welch')

    else:
            
        dat = simres[0]['e']
        nchan = dat.shape[1]# meg.shape[0]
        ch_names = list(dat.columns.values.astype(str))
        ch_types = ['mag' for _ in range(nchan)]
        raw = RawArray(dat.T/1e9, create_info(ch_names,1e3/(dat.index[1] - dat.index[0]),ch_types),verbose=False)
        raw.filter(fmin, fmax)
        # raw.plot_psd(fmin=fmin,fmax=fmax,average=False,ax=a);

        tmin = 0; tmax = raw.times[-1]
        n_fft = min(np.diff(raw.time_as_index([tmin, tmax]))[0] + 1, 2048) # probably will return 2048
        psds,freqs = psd_welch(raw,n_fft=n_fft,fmin=fmin,fmax=fmax,verbose=False)
        df_psds = pd.DataFrame(psds.T,index=freqs)#blah[0].T,index=blah[1])
        df_psds.loc[:fmax].plot(legend=False,logy=True,c='k', alpha=0.05,linewidth=0.5,ax=a)
        a.set_title('welch (mne)')
        
    
    
    a = ax[2][0]
    df = simres[2]['e']#.loc[:80]
    df.plot(logx=True,logy=True,ax=a,legend=False,c='k',alpha=0.5)

    pl_line = 1./df.index.values
    pl_line[pl_line=='inf'] = 0
    pl2_line = 1./(df.index.values**2)
    pl2_line[pl2_line=='inf'] = 0
    newdf = pd.DataFrame([pl_line,pl2_line],index=['1/f', '1/f^2'], columns=df.index).T
    newdf.plot(logx=True,logy=True,ax=a)
    a.set_title('Log power')
    #df.plot(logx=True,logy=True,ax=a)
    a.set_ylim([10E-12,10E0])


    a = ax[2][1]
    if singlenode==True:
        a.axis('off'); a.grid('off')
    else:
        sns.heatmap(simres[0]['e'].corr(),ax=a,cmap='RdBu_r',vmin=-1,vmax=1,
                    mask=np.eye(simres[0]['e'].corr().shape[0]))# ,vmax=0.3)
        a.set_title('pearson correlations')
                
    a = ax[3][0]
    if singlenode==True:
        a.axis('off'); a.grid('off')
    else:
        sns.heatmap(simres[1]['e'].loc[:100].iloc[::-1],ax=a)
        a.set_title('power over nodes')
        a.set_xlabel('node #')
        a.set_ylabel('freq')
        
    a = ax[3][1]
    if singlenode==True:
        a.axis('off'); a.grid('off')
    else:
        simres[1].idxmax().unstack(0)['e'].plot(kind='bar',ax=a)
        a.set_title('dominant freqs')
        a.set_xlabel('node #')
        a.set_xticklabels('')
        a.set_ylabel('freq')

    plt.tight_layout()

    
    if fname:
        plt.savefig(fname, bbox_inches='tight', transparent=True)
        plt.close()
    


def plot_surface_mpl(vtx,tri,data=None,rm=None,reorient='tvb',view='superior',
                     shaded=False,ax=None,figsize=(6,4), title=None,
                     lthr=None,uthr=None, nz_thr = 1E-20,
                     shade_kwargs = {'edgecolors': 'k', 'linewidth': 0.1,
                                     'alpha': None, 'cmap': 'coolwarm',
                                     'vmin': None, 'vmax': None}):
                        
  r"""Plot surfaces, surface patterns, and region patterns with matplotlib
    
  This is a general-use function for neuroimaging surface-based data, and 
  does not necessarily require construction of or interaction with tvb 
  datatypes. 

  See also:  plot_surface_mpl_mv



  Parameters
  ----------
  
  vtx           : N vertices x 3 array of surface vertex xyz coordinates 

  tri           : N faces x 3 array of surface faces

  data          : array of numbers to colour surface with. Can be either 
                  a pattern across surface vertices (N vertices x 1 array),
                  or a pattern across the surface's region mapping 
                  (N regions x 1 array), in which case the region mapping 
                  bust also be given as an argument. 
                  
  rm            : region mapping - N vertices x 1 array with (up to) N 
                  regions unique values; each element specifies which 
                  region the corresponding surface vertex is mapped to 

  reorient      : modify the vertex coordinate frame and/or orientation 
                  so that the same default rotations can subsequently be 
                  used for image views. The standard coordinate frame is 
                  xyz; i.e. first,second,third axis = left-right, 
                  front-back, and up-down, respectively. The standard 
                  starting orientation is axial view; i.e. looking down on
                  the brain in the x-y plane.
                  
                  Options: 

                    tvb (default)   : swaps the first 2 axes and applies a rotation
                                              
                    fs              : for the standard freesurfer (RAS) orientation; 
                                      e.g. fsaverage lh.orig. 
                                      No transformations needed for this; so is 
                                      gives same result as reorient=None

  view          : specify viewing angle. 
  
                  This can be done in one of two ways: by specifying a string 
                  corresponding to a standard viewing angle, or by providing 
                  a tuple or list of tuples detailing exact rotations to apply 
                  around each axis. 
                  
                  Standard view options are:
    
                  lh_lat / lh_med / rh_lat / rh_med / 
                  superior / inferior / posterior / anterior

                  (Note: if the surface contains both hemispheres, then medial 
                   surfaces will not be visible, so e.g. 'rh_med' will look the 
                   same as 'lh_lat')
                   
                  Arbitrary rotations can be specied by a tuple or a list of 
                  tuples, each with two elements, the first defining the axis 
                  to rotate around [0,1,2], the second specifying the angle in 
                  degrees. When a list is given the rotations are applied 
                  sequentially in the order given. 
                  
                  Example: rotations = [(0,45),(1,-45)] applies 45 degrees 
                  rotation around the first axis, followed by 45 degrees rotate 
                  around the second axis. 

  lthr/uthr     : lower/upper thresholds - set to zero any datapoints below / 
                  above these values
  
  nz_thr        : near-zero threshold - set to zero all datapoints with absolute 
                  values smaller than this number. Default is a very small 
                  number (1E-20), which unless your data has very small numbers, 
                  will only mask out actual zeros. 

  shade_kwargs  : dictionary specifiying shading options

                  Most relevant options (see matplotlib 'tripcolor' for full details):
                  
                    - 'shading'        (either 'gourand' or omit; 
                                        default is 'flat')
                    - 'edgecolors'     'k' = black is probably best
                    - 'linewidth'      0.1 works well; note that the visual 
                                       effect of this will depend on both the 
                                       surface density and the figure size 
                    - 'cmap'           colormap
                    - 'vmin'/'vmax'    scale colormap to these values
                    - 'alpha'          surface opacity
                  
  ax            : figure axis
  
  figsize       : figure size (ignore if ax provided)
  
  title         : text string to place above figure
  
  
  
                  
  Usage
  -----
       

  Basic freesurfer example:

  import nibabel as nib
  vtx,tri = nib.freesurfer.read_geometry('subjects/fsaverage/surf/lh.orig')
  plot_surface_mpl(vtx,tri,view='lh_lat',reorient='fs')



  Basic tvb example:
  
  ctx = cortex.Cortex.from_file(source_file = ctx_file,
                                region_mapping_file =rm_file)
  vtx,tri,rm = ctx.vertices,ctx.triangles,ctx.region_mapping
  conn = connectivity.Connectivity.from_file(conn_file); conn.configure()
  isrh_reg = conn.is_right_hemisphere(range(conn.number_of_regions))
  isrh_vtx = np.array([isrh_reg[r] for r in rm])
  dat = conn.tract_lengths[:,5]

  plot_surface_mpl(vtx=vtx,tri=tri,rm=rm,data=dat,view='inferior',title='inferior')

  fig, ax = plt.subplots()
  plot_surface_mpl(vtx=vtx,tri=tri,rm=rm,data=dat, view=[(0,-90),(1,55)],ax=ax,
                   title='lh angle',shade_kwargs={'shading': 'gouraud', 'cmap': 'rainbow'})

   
  """
    
  # Copy things to make sure we don't modify things 
  # in the namespace inadvertently. 
    
  vtx,tri = vtx.copy(),tri.copy()
  if data is not None: data = data.copy()

  # 1. Set the viewing angle 
  
  if reorient == 'tvb':
    # The tvb default brain has coordinates in the order 
    # yxz for some reason. So first change that:   
    vtx = np.array([vtx[:,1],vtx[:,0],vtx[:,2]]).T.copy()
    
    # Also need to reflect in the x axis
    vtx[:,0]*=-1

  # (reorient == 'fs' is same as reorient=None; so not strictly needed
  #  but is included for clarity)
   


  # ...get rotations for standard view options
    
  if   view == 'lh_lat'    : rots =  [(0,-90),(1,90)  ]
  elif view == 'lh_med'    : rots =  [(0,-90),(1,-90) ] 
  elif view == 'rh_lat'    : rots =  [(0,-90),(1,-90) ]
  elif view == 'rh_med'    : rots =  [(0,-90),(1,90)  ]
  elif view == 'superior'  : rots =   None
  elif view == 'inferior'  : rots =   (1,180)
  elif view == 'anterior'  : rots =   (0,-90)
  elif view == 'posterior' : rots =  [(0, -90),(1,180)]
  elif (type(view) == tuple) or (type(view) == list): rots = view 

  # (rh_lat is the default 'view' argument because no rotations are 
  #  for that one; so if no view is specified when the function is called, 
  #  the 'rh_lat' option is chose here and the surface is shown 'as is' 
                            
                            
  # ...apply rotations                          
     
  if rots is None: rotmat = np.eye(3)
  else:            rotmat = get_combined_rotation_matrix(rots)
  vtx = np.dot(vtx,rotmat)

                                    
      
  # 2. Sort out the data
                                    
                                    
  # ...if no data is given, plot a vector of 1s. 
  #    if using region data, create corresponding surface vector 
  if data is None: 
    data = np.ones(vtx.shape[0]) 
  elif data.shape[0] != vtx.shape[0]: 
    data = np.array([data[r] for r in rm])
    
  # ...apply thresholds
  if uthr: data *= (data < uthr)
  if lthr: data *= (data > lthr)
  data *= (np.abs(data) > nz_thr)

                                    
  # 3. Create the surface triangulation object 
  
  x,y,z = vtx.T
  tx,ty,tz = vtx[tri].mean(axis=1).T
  tr = Triangulation(x,y,tri[np.argsort(tz)])
                
  # 4. Make the figure 

  if ax is None: fig, ax = plt.subplots(figsize=figsize)  
  
  #if shade = 'gouraud': shade_opts['shade'] = 
  tc = ax.tripcolor(tr, np.squeeze(data), **shade_kwargs)
                        
  ax.set_aspect('equal')
  ax.axis('off')
    
  if title is not None: ax.set_title(title)
        
        
        
        
def plot_surface_mpl_mv(vtx=None,tri=None,data=None,rm=None,hemi=None,   # Option 1
                        vtx_lh=None,tri_lh=None,data_lh=None,rm_lh=None, # Option 2
                        vtx_rh=None,tri_rh=None,data_rh=None,rm_rh=None,
                        title=None,**kwargs):

  r"""Convenience wrapper on plot_surface_mpl for multiple views 
   
  This function calls plot_surface_mpl five times to give a complete 
  picture of a surface- or region-based spatial pattern. 

  As with plot_surface_mpl, this function is written so as to be 
  generally usable with neuroimaging surface-based data, and does not 
  require construction of of interaction with tvb datatype objects. 

  In order for the medial surfaces to be displayed properly, it is 
  necessary to separate the left and right hemispheres. This can be 
  done in one of two ways: 

  1. Provide single arrays for vertices, faces, data, and 
     region mappings, and addition provide arrays of indices for 
     each of these (vtx_inds,tr_inds,rm_inds) with 0/False 
     indicating left hemisphere vertices/faces/regions, and 1/True 
     indicating right hemisphere. 

     Note: this requires that 

  2. Provide separate vertices,faces,data,and region mappings for 
     each hemisphere (vtx_lh,tri_lh; vtx_rh,tri_rh,etc...)


 
  Parameters
  ----------

  (see also plot_surface_mpl parameters info for more details)

  (Option 1)

  vtx               :  surface vertices
 
  tri               : surface faces

  data              : spatial pattern to plot

  rm                : surface vertex to region mapping

  hemi              : hemisphere labels for each vertex
                      (1/True = right, 0/False = left) - 
      

  OR

  (Option 2)

  vtx_lh            : left hemisphere surface_vertices
  vtx_rh            : right ``      ``    ``     ``
  
  tri_lh            : left hemisphere surface faces 
  tri_rh            : right ``      ``    ``     ``

  data_lh          : left hemisphere surface_vertices
  data_rh          : right ``      ``    ``     ``

  rm_lh            : left hemisphere region_mapping
  rm_rh            : right ``      ``    ``     ``


  title            : title to show above middle plot
 
  kwargs           : additional tripcolor kwargs; see plot_surface_mpl

 

  Examples
  ----------

  # TVB default data

  # Plot one column of the region-based tract lengths 
  # connectivity matrix. The corresponding region is 
  # right auditory cortex ('rA1')

  ctx = cortex.Cortex.from_file(source_file = ctx_file,
                                region_mapping_file =rm_file)
  vtx,tri,rm = ctx.vertices,ctx.triangles,ctx.region_mapping
  conn = connectivity.Connectivity.from_file(conn_file); conn.configure()
  isrh_reg = conn.is_right_hemisphere(range(conn.number_of_regions))
  isrh_vtx = np.array([isrh_reg[r] for r in rm])
  dat = conn.tract_lengths[:,5]

  plot_surface_mpl_mv(vtx=vtx,tri=tri,rm=rm,data=dat,
                      hemi=isrh_vtx,title=u'rA1 \ntract length')

  plot_surface_mpl_mv(vtx=vtx,tri=tri,rm=rm,data=dat,
                    hemi=isrh_vtx,title=u'rA1 \ntract length',
                    shade_kwargs = {'shading': 'gouraud',
                                    'cmap': 'rainbow'}) 


  """
   

 
  if vtx is not None:                                    # Option 1
    tri_hemi = hemi[tri].any(axis=1)
    tri_lh,tri_rh = tri[tri_hemi==0],tri[tri_hemi==1]
  elif vtx_lh is not None:                               # Option 2
    vtx = np.vstack([vtx_lh,vtx_rh])
    tri = np.vstack([tri_lh,tri_rh+tri_lh.max()+1])

  if data_lh is not None:                                # Option 2
    data = np.hstack([data_lh,data_rh])
    
  if rm_lh is not None:                                  # Option 2 
    rm = np.hstack([rm_lh,rm_rh + rm_lh.max() + 1])
    
 

  # 2. Now do the plots for each view

  # (Note: for the single hemispheres we only need lh/rh arrays for the 
  #  faces (tri); the full vertices, region mapping, and data arrays
  #  can be given as arguments, they just won't be shown if they aren't 
  #  connected by the faces in tri )
  
  # LH lateral
  plot_surface_mpl(vtx,tri_lh,data=data,rm=rm,view='lh_lat',
                   ax=subplot(2,3,1),**kwargs)
    
  # LH medial
  plot_surface_mpl(vtx,tri_lh, data=data,rm=rm,view='lh_med',
                   ax=subplot(2,3,4),**kwargs)
    
  # RH lateral
  plot_surface_mpl(vtx,tri_rh, data=data,rm=rm,view='rh_lat',
                   ax=subplot(2,3,3),**kwargs)
    
  # RH medial
  plot_surface_mpl(vtx,tri_rh, data=data,rm=rm,view='rh_med',
                   ax=subplot(2,3,6),**kwargs)
    
  # Both superior
  plot_surface_mpl(vtx,tri, data=data,rm=rm,view='superior',
                   ax=subplot(1,3,2),title=title,**kwargs)
    
  plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0,
                      top=1.0, wspace=0, hspace=0)    
    
    
    
    

def get_combined_rotation_matrix(rotations):
  '''Return a combined rotation matrix from a dictionary of rotations around 
     the x,y,or z axes'''
  rotmat = np.eye(3)
    
  if type(rotations) is tuple: rotations = [rotations] 
  for r in rotations:
    newrot = get_rotation_matrix(r[0],r[1])
    rotmat = np.dot(rotmat,newrot)
  return rotmat



def get_rotation_matrix(rotation_axis, deg):

  '''Return rotation matrix in the x,y,or z plane'''



  # (note make deg minus to change from anticlockwise to clockwise rotation)
  th = -deg * (pi/180) # convert degrees to radians

  if rotation_axis == 0:
    return np.array( [[    1,         0,         0    ],
                      [    0,      cos(th),   -sin(th)],
                      [    0,      sin(th),    cos(th)]])
  elif rotation_axis ==1:
    return np.array( [[   cos(th),    0,        sin(th)],
                      [    0,         1,          0    ],
                      [  -sin(th),    0,        cos(th)]])
  elif rotation_axis ==2:
    return np.array([[   cos(th),  -sin(th),     0    ],
                     [    sin(th),   cos(th),     0   ],
                     [     0,         0,          1   ]])    
    
    

def plot_spectral_connectivity(coh,plv,fig=None,ax=None,fname=None,vmin_alpha=0,vmax_alpha=0.5,
                              vmin_beta=0,vmax_beta=0.5,vmin_gamma=0,vmax_gamma=0.5):
    
    coh_alpha = coh.query('freq>=8 & freq<=12').unstack('region').mean().unstack('region')
    coh_beta = coh.query('freq>12 & freq<30').unstack('region').mean().unstack('region')
    coh_gamma = coh.query('freq>=30 & freq<=100').unstack('region').mean().unstack('region')

    plv_alpha = plv.query('freq>=8 & freq<=12').unstack('region').mean().unstack('region')
    plv_beta = plv.query('freq>12 & freq<30').unstack('region').mean().unstack('region')
    plv_gamma = plv.query('freq>=30 & freq<=100').unstack('region').mean().unstack('region')


    if ax == None:
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(12,6))


    n_nodes = coh_alpha.shape[0]
    
    # Plot coherence
    a = ax[0][0]; a.set_title('COH - alpha')
    sns.heatmap(coh_alpha,vmin=vmin_alpha,vmax=vmax_alpha,mask=np.eye(n_nodes),ax=a)

    a = ax[0][1]; a.set_title('COH - beta')
    sns.heatmap(coh_beta,vmin=vmin_beta,vmax=vmax_beta,mask=np.eye(n_nodes),ax=a)

    a = ax[0][2]; a.set_title('COH - gamma')
    sns.heatmap(coh_gamma,vmin=vmin_gamma,vmax=vmax_gamma,mask=np.eye(n_nodes),ax=a)
    
    # Plot phase plocking
    a = ax[1][0]; a.set_title('PLV - alpha')
    sns.heatmap(plv_alpha,vmin=vmin_alpha,vmax=vmax_alpha,mask=np.eye(n_nodes),ax=a)

    a = ax[1][1]; a.set_title('PLV - beta')
    sns.heatmap(plv_beta,vmin=vmin_beta,vmax=vmax_beta,mask=np.eye(n_nodes),ax=a)

    a = ax[1][2]; a.set_title('PLV - gamma')
    sns.heatmap(plv_gamma,vmin=vmin_gamma,vmax=vmax_gamma,mask=np.eye(n_nodes),ax=a)

    plt.tight_layout()


    if fname != None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)
        
    

def make_brain_pic(dat,rm,vtx,tri,outfile='test_pic.png',kws=None,cmap=None,close_fig=True,reorient='tvb',
                      cb1_label = 'Amplitude',cb1_fontsize=20,annot=None,annot_fs=25,figsize=(10,3)):
    
    if cmap is None: 
        cmap = cm.RdBu_r # Reds
        cmap.set_under(color='w')

    if kws is None:
        kws = {'edgecolors': 'k', 'vmin': -0.5,'vmax': 0.5, 'cmap': cmap, 
               'alpha': None, 'linewidth': 0.01}

    rm_dat = dat[rm]
        
    fig, ax = plt.subplots(ncols=3, figsize=figsize)
    plot_surface_mpl(vtx=vtx,tri=tri,data=rm_dat,shade_kwargs=kws,view='lh_lat',ax=ax[0],reorient=reorient);
    plot_surface_mpl(vtx=vtx,tri=tri,data=rm_dat,shade_kwargs=kws,view='rh_lat',ax=ax[2],reorient=reorient);
    plot_surface_mpl(vtx=vtx,tri=tri,data=rm_dat,shade_kwargs=kws,view='superior',ax=ax[1],reorient=reorient);

    
    

    
    
    ax1 = fig.add_axes([0.2, -0.1, 0.6, 0.1])
    norm = mpl.colors.Normalize(vmin=kws['vmin'], vmax=kws['vmax'])
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,norm=norm,orientation='horizontal')
    cb1.set_label(cb1_label,fontsize=cb1_fontsize)
    
    if annot is not None:
        #ax[2].text(0.5,0.5,annot,fontdict={'fontsize':30})
        
        ax[2].text(0.8, -0.2,annot, horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax[2].transAxes,
        fontsize=annot_fs)
        
        
    plt.tight_layout()
    
    plt.savefig(outfile, transparent=True,bbox_inches='tight')
    
    if close_fig: plt.close()
        
    return fig,ax



def make_brain_movie(dat,rm,vtx,tri,png_pfx='test_file',#out_file = 'test_clip.mp4',
                     interval = 10,kws=None,cmap=None,fps=5.,png_files = [],figsize=(10,3)):#,#                     combine_pics=True):
                    
    
    timeslices = np.arange(0,dat.shape[0],interval)

    if png_files is not []:
        
        for t_it in timeslices:

            t = dat.index.values[t_it]
            
            
            #annot = '1.0s' %t
            annot = '%0.3d ms' % t
            
            png_file = png_pfx + '_%1.3d.png' %t

            dat_t = dat.values[t_it,:]
                
            stuff = make_brain_pic(dat=dat_t, rm=rm,vtx=vtx,tri=tri,close_fig=True,figsize=figsize,
                                  outfile = png_file,kws=kws,cmap=cmap,annot=annot)
        
            png_files.append(png_file)
            
    imseq = ImageSequenceClip(png_files,fps=fps) 
    
    return imseq,png_files



    
    
    
"""
Misc utility functions
"""

        
        
def get_weights_and_delays(conn_dist=None,delays_dist=None,n_nodes=None,cw_mn=None,cw_std=None,
                           cd_mn=None,cd_std=None,tau=0):#None):

    # Uniform weights:
    #     weights,delays = get_weights_and_delays('uniform', 'uniform',10,
    #                         cw_mn=10,cd_mn=10)
    
    # Initialize weights
    if conn_dist == 'rand':
            weights = rand(n_nodes,n_nodes)*cw_mn
    elif conn_dist == 'uniform': 
            weights = ones([n_nodes,n_nodes])*cw_mn
    else:
        weights = normal(cw_mn,cw_std,[n_nodes,n_nodes])
                        
    # Initialize delays        
    if delays_dist == 'rand': 
        delays = rand(n_nodes,n_nodes) * cd_mn    
    elif delays_dist == 'uniform': 
        delays = ones([n_nodes,n_nodes]) * cd_mn
    else:
        delays = normal(cd_mn,cd_std,[n_nodes,n_nodes])
 
    return weights,delays






