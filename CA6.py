
# coding: utf-8

# Mingrui Liu Music Assignment

# In[72]:

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from IPython.display import Audio, Image
import scipy.signal as sig
import matplotlib as mpl
get_ipython().magic('matplotlib inline')


# In[73]:

H = 0
Fs = 8000
C3, D3, E3, F3, G3, A3, B3 = 48, 50, 52, 53, 55, 57, 59
C, D, E, F, G, A, B = 60, 62, 64, 65, 67, 69, 71
C5, D5, E5, F5, G5, A5, B5 = 72, 74, 76, 77, 79, 81, 83 
scale = [C3, D3, E3, F3, G3, A3, B3, C, D, E, F, G, A, B, C5, D5, E5, F5, G5, A5, B5 ]


# In[74]:

def trap_env(t, dur=1, up=0.1, down=0.3):
    condlist = [t<up, t<dur-down, True]
    choicelist = [t/up, 1, (dur-t)/down]
    return np.select(condlist, choicelist)


# In[75]:

def string_env(t, dur=1):
    env = (1-np.exp(-80*t)*np.exp(-8*t))
    return env/np.max(env)


# Sky City

# In[76]:

Sky_City = [ (H, 3/5), (H, 3/5), (H, 3/5), (A, 3/10), (B, 3/10), (C5, 9/10),
        (B, 3/10), (C5, 3/5), (E5, 3/5), (B, 6/5), (H, 3/5), (G, 3/5),
        (A, 9/10), (G, 3/10), (A, 3/5), (C5, 3/5), (G, 6/5), (H, 3/5),
        (F, 3/10), (E, 3/10), (F, 9/10), (E, 3/10), (F, 3/5), (C5, 3/5),
        (E, 6/5), (H, 3/10),(C5, 3/10),(C5, 3/10), (C5, 3/10), (B, 9/10),
        (F, 3/10), (F, 3/5), (B, 3/5), (B, 6/5), (H, 3/5), (A, 3/10), (B, 3/10),
        (C5, 9/10), (B, 3/10), (C5, 3/5), (E5, 3/5), (B, 6/5), (H, 3/5), (E, 3/10),
        (E, 3/10), (A, 9/10), (G, 3/10), (A, 3/5), (C5, 3/5), (G, 6/5), (H, 3/5),
        (E, 3/5), (F, 3/5), (C5, 3/10), (B, 9/10), (C5, 3/5), (D5, 3/5), (E5, 3/10),
        (C5, 3/2), (C5, 3/10), (B, 3/10), (A, 3/5), (B, 3/5), (G, 3/5), (A, 9/5),
        (C5, 3/10), (D5, 3/10), (E5, 9/10), (D5, 3/10), (E5, 3/5), (G5, 3/5), (D5, 6/5),
        (H, 3/5), (G, 3/10), (G, 3/10), (C5, 9/10), (B, 3/10), (C5, 3/5), (E5, 3/5),
        (E5, 9/5), (H, 3/5), (A, 3/10), (B, 3/10), (C5, 3/5), (B, 3/10), (C5, 3/10), (D5, 3/5),
        (C5, 9/10), (G, 3/10), (G, 6/5), (F5, 3/5), (E5, 3/5), (D5, 3/5), (C5, 3/5),
        (E5, 21/5), (E5, 3/5), (A5, 6/5), (G5, 6/5), (E5, 3/5), (D5, 3/10), (C5, 3/2), (D5, 3/5),
        (C5, 3/10), (D5, 9/10), (G5, 3/5), (E5, 9/5), (E5, 3/5), (A5, 6/5), (G5, 6/5),
        (E5, 3/5), (D5, 3/10), (C5, 3/2), (D5, 3/5), (C5, 3/10), (D5, 9/10), (B, 3/5),
        (A, 9/5), (A, 3/10), (B, 3/10), (C5, 9/10), (B, 3/10), (C5, 3/5), (E5, 3/5),
        (B, 6/5), (H, 3/5), (E, 3/5), (A, 9/10), (G, 3/10), (A, 3/5), (C5, 3/5), (G, 6/5),
        (H, 3/5), (F, 3/10), (E, 3/10), (F, 9/10), (E, 3/10), (F, 3/5), (C5, 3/5), (E, 6/5),
        (H, 3/10), (C5, 3/10), (C5, 3/10), (C5, 3/10), (B, 9/10), (F, 3/10), (F, 3/5), (B, 3/5),
        (B, 6/5), (H, 3/5), (A, 3/10), (B, 3/10), (C5, 9/10), (B, 3/10), (C5, 3/5), (E5, 3/5),
        (B, 6/5), (H, 3/5), (E, 3/10), (E, 3/10), (A, 9/10), (G, 3/10), (A, 3/5), (C5, 3/5),
        (G, 6/5), (H, 3/5), (E, 3/5), (F, 3/5), (C5, 3/10), (B, 9/10), (C5, 3/5), (D5, 3/5),
        (E5, 3/10), (C5, 3/2), (C5, 3/10), (B, 3/10), (A, 3/5), (B, 3/5), (G, 3/5), (A, 12/5),]
def playsong(song, env=trap_env, basenote=440, Fs=22500, time=1):
    sounds = []
    for note in song:
        fnum, dur = note
        t = np.linspace(0,dur*time,int(dur*time*Fs),endpoint=False)
        f = basenote * 2**((fnum-69)/12)
        sinusoid = np.sin(2*np.pi*f*t)
        sounds.append(env(t,dur*time) * sinusoid)
    return np.concatenate(sounds)

Audio(playsong(Sky_City,Fs=Fs),rate=Fs)


# In[77]:

string_Sky_City = playsong(Sky_City, env=string_env, Fs = Fs)
Audio(string_Sky_City, rate = Fs)


# In[78]:

next_Sky_City = playsong(Sky_City, env=string_env, basenote = 880, Fs = Fs)
Audio(next_Sky_City, rate=Fs)


# In[79]:

total_Sky_City = string_Sky_City + next_Sky_City
Audio(total_Sky_City, rate = Fs)


# In[80]:

def Clip(sky_city):
    clipping_sky_city = playsong(sky_city, env=string_env, Fs = Fs)
    clipped = np.clip(string_Sky_City, -0.6,0.6)/0.6
    return clipped

Audio(Clip(Sky_City), rate = Fs)


# Tremolo

# In[81]:

def Tremolo(sky_city):
    depth = 0.6
    tremfreq = 8
    Fs = 8000
    basenote = 880
    sounds = []
    for note in Sky_City:
        fnum, dur = note
        t = np.linspace(0,dur,int(dur*Fs),endpoint=False)
        tremolo = 1 + depth*np.sin(2*np.pi*tremfreq*t)
        f = basenote * 2**((fnum-69)/12)
        sinusoid = np.sin(2*np.pi*f*t)
        sounds.append(trap_env(t,dur) * tremolo * sinusoid)
    sound = np.concatenate(sounds)
    return sound
Audio(Tremolo(Sky_City), rate = Fs)


# Harmonics

# In[82]:

def Harmonics(sky_city):
    harmonics = [1.0,0.5,0.4,0.3,0.2] #organ-like
    basenote = 880
    ks = np.arange(1,len(harmonics)+1)
    sounds = []
    for note in Sky_City:
        fnum, dur = note
        t = np.linspace(0,dur,int(dur*Fs),endpoint=False)
        f = basenote * 2**((fnum-69)/12)
        sound = harmonics @ np.sin(2*np.pi*f*np.outer(ks,t))
        sounds.append(trap_env(t,dur,down=0.1) * sound)
    sound = np.concatenate(sounds)
    Audio(sound, rate=Fs)
    return sound
Audio(Harmonics(Sky_City), rate = Fs)

