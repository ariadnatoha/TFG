#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:51:06 2022

@author: ariadnatoha
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import mpmath

X=100
Y=300
#Z=100
N=5000
co=N/X/Y*10**4 #particules/cm^3
x=[]
y=[]
#z=[]
conc=[]
for i in range(N):
    a=random.random()
    b=random.random()
    #c=random.random()
    x.append(a*X)
    y.append(b*Y)
    #z.append(c*Z)

plt.plot(x,y,'o',markersize=1)
plt.show()

dt=1 #cada pas de temps (s)
T=600 #nombre de passos de temps, CANVIAR PER OBTENIR RESULTATS COHERENTS
Tfinal=T*dt
#velocitat serà en m/s
#constants:

muo=4*np.pi*10**(-7) #N/A^2
a=0.007 #radi iman (m)
M=42.7 #Am^2/kg
mu=8.52*10**(-4) #kg/ms
rh=15*10**(-9) #m 
m=5*10**(-20) #kg 

Br=1.45
h=0.015 #alçada iman(m)
kb=1.38*10**(-23)
temp=300
mmag=1 #no ho se pero tampoc ho estic utilitzant

g=9.81 #m/s^2

Fg=m*g
xB=[]
ygradB=[]
yv=[]
temps=[]
n=0

def L(x):
    resultat=mpmath.coth(x)-1/x
    return resultat

for t in range(T):
    #plt.plot(x,y,'o',markersize=1)
    #plt.show()
    c=(N-n)/X/Y*10**4
    conc.append(c/co)
    temps.append(t*dt)
    n=0
    for i in range (N):
        if y[i]>0:
            #B=Br/2*((y[i]*10**(-4)+h)/((y[i]*10**(-4)+h)**2+a**2)**0.5-y[i]*10**(-4)/((y[i]*10**(-4))**2+a**2)*0.5)
            gradB=-Br*a**2/2*(((y[i]*10**(-4)+h)**2+a**2)**(-1.5)-((y[i]*10**(-4))**2+a**2)**(-1.5))
            #print(gradB) 
            Fm=m*M*gradB #*L(mmag*a/muo/kb/temp*B)
            #print(L(mmag*a/muo/kb/temp*B))
            v=(Fg+Fm)/(6*np.pi*mu*rh) #m/s (treballant en valor absolut)
            if t==0:
                xB.append(y[i])
                ygradB.append(gradB)
                yv.append(v)
            y[i]=y[i]-v*dt
        if y[i]<0:
            y[i]=0
        if y[i]<0.5:
            n+=1
plt.plot(xB,ygradB,'o',markersize=1)
plt.title('gradB (T/m) a cada y')
plt.show()
plt.plot(xB,yv,'o',markersize=1)
plt.title('v (m/s) a cada y')
plt.show()
plt.plot(temps,conc,'o',markersize=1)
plt.title('c/co')
plt.show()
print(n, "col·loides han arribat a la base")
plt.plot(x,y,'o',markersize=1)
plt.show()