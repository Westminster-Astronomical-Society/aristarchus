#####################################################################################################
#       __     __  ___       __   __             __      __              __          __       
#  /\  |__) | /__`  |   /\  |__) /  ` |__| |  | /__`    /  `  /\   |\/| |__)  /\  | / _` |\ | 
# /~~\ |  \ | .__/  |  /~~\ |  \ \__, |  | \__/ .__/    \__, /~~\  |  | |    /~~\ | \__> | \| 
#
#####################################################################################################
# Script: get-moon-radius.py
# Usage: $ python get-moon-radius.py <IMAGE>
#####################################################################################################
# Last update: 04-21-2014 (Jorge I. Zuluaga)
# 2014 (C) Jorge I. Zuluaga
#####################################################################################################

from matplotlib import cm,pyplot as plt
from scipy.misc import *
from sys import exit,argv
from numpy import *
from scipy.optimize import fmin
from cv2 import imread
#from matplotlib.pyplot import imread

###################################################
#ROUTINES
###################################################
def imageCentroid(imagen,threshold=50):
    height,width=imagen.shape
    cond=imagen>=threshold
    X,Y=meshgrid(arange(width),arange(height))
    xm=X[cond].mean()
    ym=Y[cond].mean()
    return xm,ym

def detectBorder(imagen,threshold=50,R=900):
    cond=imagen>=threshold
    xm,ym=imageCentroid(imagen,threshold=threshold)
    plt.close("all")
    contour=plt.contour(cond,levels=[0.0])
    border=[]
    for path in contour.collections[0].get_paths():
        for points in path.vertices:
            if abs(points[0]-xm)>R or abs(points[1]-ym)>R:
                continue
            border+=[points.tolist()]
    border=array(border)
    return xm,ym,border

###################################################
#PROCESSING IMAGES
###################################################
filename=argv[1]
basename=filename.split(".")[0]
print("Processing image %s..."%filename)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#LOAD IMAGE
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imagen=imread(filename)[:,:,0]
height,width=imagen.shape
print("Image size:",width,height)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#DETECT CENTROID AND BORDER
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def errorR(params):
    thresh=params[0]
    try:
        xm,ym,border=detectBorder(imagen,threshold=thresh)
    except:
        return 1e6
    xborder=border[:,0]
    yborder=border[:,1]
    xm=xborder.mean()
    ym=yborder.mean()
    rborder=((xborder-xm)**2+(yborder-ym)**2)**0.5
    R=rborder.mean()
    dR=rborder.std()
    print(f"\tThreshold: {thresh:.1f}, Measured Radius : {R:.4f} +/- {dR:.2f}")
    return dR

print("Minimizing uncertainties:")
solution=fmin(errorR,[80.0],xtol=0.5,maxiter=10)
thrmin=solution[0]
print("Threshold:",thrmin)

xmc,ymc,borderc=detectBorder(imagen,threshold=thrmin)
dRmin=errorR([thrmin])
rborder=((borderc[:,0]-xmc)**2+(borderc[:,1]-ymc)**2)**0.5
Rmin=rborder.mean()
print("Radius at minimum uncertainty: %.4f +/- %.2f"%(Rmin,dRmin))

###################################################
#PLOT IMAGE
###################################################
xm,ym,borderm=detectBorder(imagen,threshold=20)
xm,ym,borderM=detectBorder(imagen,threshold=120)

fig=plt.figure(figsize=(8,8))
ax=fig.add_axes([0.0,0.0,1.0,1.0])
plt.imshow(imagen,cmap=cm.gray)
border=borderc
plt.plot(border[::,0],border[::,1],'c.',markersize=0.5)
border=borderm
plt.plot(border[::,0],border[::,1],'y.',markersize=0.5)
border=borderM
plt.plot(border[::,0],border[::,1],'r.',markersize=0.5)
plt.xlim((xmc-1.2*Rmin,xmc+1.2*Rmin));plt.ylim((ymc-1.2*Rmin,ymc+1.2*Rmin))
plt.savefig(basename+"-border.png")

