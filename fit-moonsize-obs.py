#####################################################################################################
#       __     __  ___       __   __             __      __              __          __       
#  /\  |__) | /__`  |   /\  |__) /  ` |__| |  | /__`    /  `  /\   |\/| |__)  /\  | / _` |\ | 
# /~~\ |  \ | .__/  |  /~~\ |  \ \__, |  | \__/ .__/    \__, /~~\  |  | |    /~~\ | \__> | \| 
#
#####################################################################################################
# Script: fit-moonsize-obs.py
# Usage: $ python get-moon-radius.py
#####################################################################################################
# Last update: 04-21-2014 (Jorge I. Zuluaga)
# 2014 (C) Jorge I. Zuluaga
#####################################################################################################

from matplotlib import cm,pyplot as plt
from scipy.misc import *
import sys
from numpy import *
from scipy.optimize import fmin
from scipy.stats import chi2
from scipy.integrate import quad as integ
import os
exit=sys.exit
argv=sys.argv

###################################################
#CONSTANTS
###################################################
RE=1.0
DEG=pi/180

###################################################
#INPUT PARAMETERS
###################################################
OBSERVER_LATITUDE=6.127910
MONTECARLO_POINTS=10000
FIT_TEST=200

#MEASURED INITIAL ELEVATION
H_INI=11.0
D_HINI=2.0

#MEASURED INITIAL AZIMUTH
A_INI=101.0
D_AINI=2.0

#JUST READ THE MONTE CARLO RESULTS
JUST_READ=True if os.path.isfile('MC.dat') else False

#RANGE OF THE APPARENT DIAMETER PARAMETER
MIN_FO=100*2
MAX_FO=1000*2

###################################################
#ESTIMATED DECLINATION AND HOUR ANGLE
###################################################
D=60.0
hs=random.normal(H_INI,D_HINI,size=100)
As=random.normal(A_INI,D_AINI,size=100)
dh=arcsin(RE/D*cos(hs*DEG))/DEG
ht=hs+dh
delta=arcsin(sin(OBSERVER_LATITUDE*DEG)*sin(ht*DEG)+
             cos(OBSERVER_LATITUDE)*cos(ht*DEG)*cos(As*DEG))/DEG
Ho=arccos((sin(ht*DEG)-sin(delta*DEG)*sin(OBSERVER_LATITUDE))/
          (cos(delta*DEG)*cos(OBSERVER_LATITUDE)))/DEG

Ho=(360-Ho)/15

###################################################
#ESTIMATED DECLINATION AND HOUR ANGLE
###################################################
MIN_DELTA=delta.min()
MAX_DELTA=delta.max()
MIN_HO=Ho.min()
MAX_HO=Ho.max()

print("Range of delta: %e - %e"%(MIN_DELTA,MAX_DELTA))
print("Range of hour angle: %e - %e"%(MIN_HO,MAX_HO))

###################################################
#ROUTINES
###################################################
def pvalue(chisquare,n,p):
    nu=n-p
    xmax=max(10*chisquare,100*nu)
    chinu=lambda x:chi2.pdf(x,nu)
    qval,erro=integ(chinu,0,chisquare)
    pval=1-qval
    return abs(pval)

def randRange(xmin,xmax):return xmin+(xmax-xmin)*random.rand()

def apparentSizeDistance(dt,param):
    D=param[0]
    delta=param[1]
    Ho=param[2]
    fo=param[3]

    Hi=Ho*15+dt*14.492053702994191
    hi=arcsin(sin(delta*DEG)*sin(OBSERVER_LATITUDE*DEG)+
              cos(delta*DEG)*cos(OBSERVER_LATITUDE*DEG)*cos(Hi*DEG))/DEG
    di=(D**2+RE**2-2*D*RE*sin(hi*DEG))**0.5
    Ri=fo*60/di

    #print(fo*60)
    #print(D,delta,Ho,fo,Hi,hi,di,Ri)
    #exit(0)
    return Ri,di

###################################################
#DATA
###################################################
f=open("img-data.dat")
ts=[]
Rs=[]
dRs=[]
for line in f:
    if "#" in line:continue
    fields=line.split()
    tstr=fields[0].split(":")
    ts+=[int(tstr[0])+int(tstr[1])/60.+int(tstr[2])/3600.]
    Rs+=[float(fields[1])]
    dRs+=[float(fields[2])]    
numpoints=len(ts)
ts=array(ts)
Rs=array(Rs)
dRs=array(dRs)

###################################################
#FITTING DATA
###################################################
def moonChiSquare(param,verbose=True,**args):
    D=param[0]
    delta=param[1]
    Ho=param[2]
    fo=param[3]

    if (Ho<MIN_HO or Ho>MAX_HO) or (delta<MIN_DELTA or delta>MAX_DELTA) or (fo<MIN_FO or fo>MAX_FO):
        if verbose:
            print("Out of ranges...")
        return 1000

    chisquare=0
    Rit=[]
    for i in range(0,numpoints):
        #THEORETICAL MODEL
        Ri=apparentSizeDistance(ts[i]-ts[0],param)[0]
        Rit+=[Ri]

        #OBSERVED VALUES
        Riobs=Rs[i]
        dRiobs=dRs[i]

        #CHISQUARE
        chisquare+=(Ri-Riobs)**2/dRiobs**2

    if verbose:
        print("\tTesting (D=%e,delta=%e,Ho=%e,fo=%e) : chisquare = %.17e"%(D,delta,Ho,fo,chisquare))
        
    return chisquare

print("Fitting data...")
itest=0
chisqmin=1E100
while True:
    sys.stdout.write('\r')
    sys.stdout.write("\t%.1f%%"%(100*float(itest)/FIT_TEST))
    sys.stdout.flush()

    D=randRange(50,70)
    Ho=randRange(MIN_HO,MAX_HO)
    delta=randRange(MIN_DELTA,MAX_DELTA)
    fo=randRange(MIN_FO,MAX_FO)
    guess=[D,delta,Ho,fo]
    output=fmin(moonChiSquare,guess,args=(False,),full_output=True,disp=False)
    solucion=output[0]
    chisq=output[1]
    pval=pvalue(chisq,numpoints,4)
    if chisq<chisqmin:
        solucionmin=solucion
        chisqmin=chisq
    itest+=1
    if itest>FIT_TEST or JUST_READ:break
print
print("Done.")

chisq=moonChiSquare(solucionmin,verbose=True)
pval=pvalue(chisq,numpoints,4)
print("Best-fit:")
print("\tD = %e"%solucionmin[0])
print("\tdelta = %e"%solucionmin[1])
print("\tHo = %e"%solucionmin[2])
print("\tfo = %e"%solucionmin[3])
print("\tChi-square = %.17e"%chisq)
print("\tP-value = %e"%pval)
print()

###################################################
#PLOT APPARENT SIZES
###################################################
fig=plt.figure()
plt.errorbar(ts-ts[0],Rs,yerr=dRs,fmt='o',label='Measurements')
dts=linspace(0.95*ts[0],1.05*ts[-1],100)-ts[0]
#plt.plot(dts,apparentSizeDistance(dts,solucionmin)[0],'k-',label='Best-fit')
plt.xlabel("Time since the first observation (h)")
plt.ylabel("Moon apparent size (px)")


###################################################
#MONTECARLO CONFIDENCE LIMITS CALCULATION
###################################################
if not JUST_READ:
    D,delta,Ho,fo=solucionmin
    Dmin=D-10.0
    Dmax=D+10.0
    deltamin=delta-5.0
    deltamax=delta+5.0
    Homin=Ho-0.5
    Homax=Ho+0.5
    fomin=fo-50
    fomax=fo+50
    
    imc=0
    Ds=[]
    deltas=[]
    Hos=[]
    fos=[]
    
    print("Performing Monte Carlo Confidence level analysis...")
    fmc=open("MC.dat","w")
    while True:
        sys.stdout.write('\r')
        sys.stdout.write("\t%.1f%%"%(100*float(imc)/MONTECARLO_POINTS))
        sys.stdout.flush()
        
        Dr=randRange(Dmin,Dmax)
        deltar=randRange(deltamin,deltamax)
        Hor=randRange(Homin,Homax)
        fora=randRange(fomin,fomax)
        solucion=[Dr,deltar,Hor,fora]
        chisq=moonChiSquare(solucion,verbose=False)
        pval=pvalue(chisq,numpoints,4)
        
        if pval>0.05 and pval<0.95:
            #print "P-value (%d): %e"%(imc,pval)
            Ds+=[Dr]
            deltas+=[deltar]
            Hos+=[Hor]
            fos+=[fora]
            fmc.write("%e %e %e %e\t%e %e\n"%(Dr,deltar,Hor,fora,chisq,pval))
            fmc.flush()

        imc+=1
        if imc>MONTECARLO_POINTS:break

    Ds=array(Ds);deltas=array(deltas);Hos=array(Hos);fos=array(fos)
    fmc.close()
    print
    print("Done.")

data=loadtxt("MC.dat")
Ds=data[:,0]
deltas=data[:,1]
Hos=data[:,2]
fos=data[:,3]
numdata=len(Ds)
for i in range(numdata):
    Dr=Ds[i]
    deltar=deltas[i]
    Hor=Hos[i]
    fora=fos[i]
    plt.plot(dts,apparentSizeDistance(dts,[Dr,deltar,Hor,fora])[0],'-',color=cm.gray(0.8),zorder=-10)

print("Number of Monte Carlo points:",numdata)
print("D:")
print("\tMin:",Ds.min())
print("\tMax:",Ds.max())
print("\tMean:",Ds.mean())
print("\tStd.Dev.:",Ds.std())
minD=Ds.mean()-Ds.min()
maxD=Ds.max()-Ds.mean()

print("delta:")
print("\tMin:", deltas.min())
print("\tMax:", deltas.max())
print("\tMean:", deltas.mean())
print("\tStd.Dev.:", deltas.std())

print("Ho:")
print("\tMin:", Hos.min())
print("\tMax:", Hos.max())
print("\tMean:", Hos.mean())
print("\tStd.Dev.:", Hos.std())

print("fo:")
print("\tMin:", fos.min())
print("\tMax:", fos.max())
print("\tMean:", fos.mean())
print("\tStd.Dev.:", fos.std())

Dmean = Ds.mean()
deltamean=deltas.mean()
Homean=Hos.mean()
fomean=fos.mean()

plt.plot(dts,apparentSizeDistance(dts,[Dmean,deltamean,Homean,fomean])[0],
         '-',color=cm.gray(0.0))

###################################################
#FINISH PLOT
###################################################
plt.title("$D/R_E = %.2f^{+%.2f}_{-%.2f}$ (95%% C.L.)"%(Dmean,maxD,minD),position=(0.5,1.02))
plt.legend(loc='best')
plt.savefig("bestfit-sizes.png")

###################################################
#PLOT DISTANCES
###################################################
plt.figure()
dso=[]
ddso=[]
for i in range(0,numpoints):
    dso+=[60*fomean/Rs[i]]
    ddso+=[dso[i]*(dRs[i]/Rs[i])]
plt.plot(dts,apparentSizeDistance(dts,[Dmean,deltamean,Homean,fomean])[1],'k-')
plt.errorbar(ts-ts[0],dso,yerr=ddso,fmt='o',label='Measurements')
plt.xlabel("Time since the first observation (h)")
plt.ylabel("Moon instantaneous distance, $d/R_E$")
plt.title("$D/R_E = %.2f^{+%.2f}_{-%.2f}$ (95%% C.L.)"%(Ds.mean(),maxD,minD),position=(0.5,1.02))
plt.legend(loc='best')
plt.savefig("bestfit-distances.png")
