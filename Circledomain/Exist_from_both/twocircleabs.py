#!/usr/bin/env python3
#crc code 

#this code this made for what happens if we have both boundarys abs 

import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
from scipy.special import i0
import random 

def d(x1,y1,x2,y2):
    dist = np.sqrt((x2-x1)**2+(y2-y1)**2)#get dist between the points
    return dist


def findxy(var0,var1,var2,var3,placeholderradius):
    r = placeholderradius # radius of which circle the particule crossed 
    
    if (var2==var0): # if the y are the same
        y = (r**2-var0**2)**0.5
        x = var0
        if (var1<0): 
            y=-y
        distleft = d(var2,var3,x,y) 
        if y > 0:
            y = y- distleft
        else:
            y = y+distleft
        
        return(x,y)
    elif(var1==var3): # if the x are the same
        y=var1
        x =  (r**2 - y**2)**0.5
        if (var0<0):
            x=-x     
        distleft = d(var2,var3,x,y)
        if x > 0:
            x = x- distleft
        else:
            x = x+ distleft
       
        return(x,y)
    else:
        
        m = (var3-var1)/(var2-var0)#getting m 
        b = var3 - m*var2 # getting b
        
        
        a1 = (1+m**2)# getting a for x 
        b1 = (2*m*b )#getting b for x
        c1 = (b**2-r**2)#getting c for x
        
       
        ay = (1/m**2 +1) #getting a for y 
        by = -((2*b)/m**2)#getting b for y 
        cy = (b**2/m**2 - r**2)#getting c fo y

        
        y1 = (-by+np.sqrt(by**2-4*ay*cy))/(2*ay) 
        y2 = (-by-np.sqrt(by**2-4*ay*cy))/(2*ay)
            
        x1 = (-b1+np.sqrt(b1**2-4*a1*c1))/(2*a1) 
        x2 = (-b1-np.sqrt(b1**2-4*a1*c1))/(2*a1)
            
        if r == .5: # if it cross the smaller radius we us the - 
            if (var0<= x1<= var2):
                x=x2
            else:
                x=x1
            if (var1<= y1<= var3):
                y=y2
            else:
                y=y1
        else: # if it cross the smaller radius we us the +

            if (var0<= x1<= var2):
                x=x1
            else:
                x=x2
            if (var1<= y1<= var3):
                y=y1
            else:
                y=y2
            
            # return the coords for int            
        return(x,y)


def k(x,y): # setting alpha 
    return 1.7260
    
def q(x,y,phi):
    theta = math.atan2(y,x)
    return(1/(2*np.pi*i0(k(x,y)))) * np.cosh(k(x,y)*np.sin(phi-theta))# when alpha is neg this should be sin 
    #return (1/(2*np.pi*i0(k(x,y)))) * np.cosh(k(x,y)*np.cos(phi-theta))


def get_angle(xcord,ycord): # for each new position we get the angle that it should go 
    angle = np.zeros(len(xcord))
    for n in range(len(xcord)):
        M = np.cosh(k(xcord[n],ycord[n]))/(2*np.pi*i0(k(xcord[n],ycord[n])))
        N = 1
        N_accept = 0
        y2_list = np.zeros(N)
        while N_accept < N:
            u = np.random.rand()
            y2 = 2*np.pi*np.random.rand(); # proposal distribution uniform.
            if u < q(xcord[n],ycord[n],y2)/M:          
                y2_list = y2
                N_accept += 1
        angle[n]+= y2 #picking a rand num from the dist
    return(angle) # return the angle 

def beta(alpha):
    return((2*alpha)/(1+alpha))


if __name__ == "__main__":
    rho = .5 
    bigr = 3
    alpha = -.25
    r = np.linspace(.51,2.99, num=10)
    mu = 1000
    v = np.sqrt(mu)
    D= v**2/(2*mu)
    rounds = 1 #how many eps 
    means = [] # blank array
    xspacefull =  r # getting the linespace 
    
    savedmeans2 = []
    actualsolution = (1/(4*D))*(-r**2+(bigr**beta(alpha)*rho**2-bigr**2*rho**beta(alpha))/(bigr**beta(alpha)-rho**beta(alpha))+r**beta(alpha)*(bigr**2-rho**2)/(bigr**beta(alpha)-rho**beta(alpha)))
    #we want a linespace of positions and then we are going graph the mean first passage time agiast the actual solution
    for xspace in xspacefull: # for loop through the linespace 
        escape_time=np.zeros(rounds)
        xnew = np.zeros(rounds)
        ynew = np.zeros(rounds)
        xtestnew = np.zeros(rounds)
        ytestnew = np.zeros(rounds)
        times = np.zeros(rounds)
        amout_capt = np.zeros(rounds)

        #getting the starting positions 
        for n in range(rounds):
            xnew[n] +=xspace
            ynew[n] +=0

        #stacking to fill in the new position
        xnew = np.vstack([xnew, np.zeros(rounds)])
        ynew = np.vstack([ynew, np.zeros(rounds)])
        xtestnew = np.vstack([xtestnew, np.zeros(rounds)])
        ytestnew = np.vstack([ytestnew, np.zeros(rounds)])
        times = np.vstack([times, np.zeros(rounds)])
        
        n= 0 

        #r = np.random.rand(rounds)
        r = np.random.rand(rounds)
        
        angle  = np.random.rand(rounds)* 2 * math.pi # make this random 

        time_diff = -1/mu *np.log(r)

        times[n+1]= times[n]+time_diff
        xnew[n+1] =xnew[n]+v*np.cos(angle)*(time_diff)
        ynew[n+1] =ynew[n]+v*np.sin(angle)*(time_diff)



        radius = np.sqrt( xnew[n+1]**2 + ynew[n+1]**2 ) 
        blanknum = 0 

        for ns in (np.where((radius >= 3) | (radius <= 0.5))[0]):
            if radius[ns]>=3:
                placeholderradius = 3
            else:
                placeholderradius = .5
            xnew[n+1][ns],ynew[n+1][ns] = findxy(xnew[n][ns],ynew[n][ns],xnew[n+1][ns],ynew[n+1][ns],placeholderradius)
            inter_time = (xnew[n+1][ns]-xnew[n][ns])/(v*np.cos(angle[ns]))
            escape_time[blanknum] += times[n][ns]+np.abs(inter_time)
            blanknum +=1

        indices = np.where((radius >= 3) | (radius <= 0.5))
        xnew = np.delete(xnew,indices,1)
        ynew = np.delete(ynew,indices,1)
        times = np.delete(times,indices,1)


        xnew= np.vstack([xnew, np.zeros(len(xnew[n+1]))])
        ynew= np.vstack([ynew, np.zeros(len(xnew[n+1]))])
        times = np.vstack([times,np.zeros(len(xnew[n+1]))])

        n=1

        while any(escape_time==0):
            angle  = get_angle(xnew[n],ynew[n])
            r = np.random.rand(len(xnew[0]))
            time_diff = -1/mu *np.log(r)
            times[n+1]= times[n]+time_diff
            xnew[n+1] =xnew[n]+v*np.cos(angle)*(time_diff)
            ynew[n+1] =ynew[n]+v*np.sin(angle)*(time_diff)
            radius = np.sqrt( xnew[n+1]**2 + ynew[n+1]**2 ) 
            for ns in (np.where((radius >= 3) | (radius <= 0.5))[0]):
                if radius[ns]>=3:
                    placeholderradius = 3
                else:
                    placeholderradius = .5
                xnew[n+1][ns],ynew[n+1][ns] = findxy(xnew[n][ns],ynew[n][ns],xnew[n+1][ns],ynew[n+1][ns],placeholderradius)
                inter_time = (xnew[n+1][ns]-xnew[n][ns])/(v*np.cos(angle[ns]))
                escape_time[blanknum] += times[n][ns]+np.abs(inter_time)
                blanknum +=1

            indices = np.where((radius >= 3) | (radius <= 0.5))
            xnew = np.delete(xnew,indices,1)
            ynew = np.delete(ynew,indices,1)
            times = np.delete(times,indices,1)

            xnew= np.vstack([xnew, np.zeros(len(xnew[n+1]))])
            ynew= np.vstack([ynew, np.zeros(len(xnew[n+1]))])
            times = np.vstack([times, np.zeros(len(xnew[n+1]))])
            n+=1

        means.append(escape_time)
        

        numc=0
        timelistfinal = sorted(escape_time)
        for x in timelistfinal:
            amout_capt[numc:] += 1 
            numc+=1 
        
        savedmeans2.append(np.mean(timelistfinal))
    #plotting all of the linspace mean first pasage time with their confident int
        
    plt.figure()
    plt.scatter(np.linspace(.51,2.99,num=10),savedmeans2)
    plt.plot(np.linspace(.51,2.99,num=10),actualsolution)
    ci = 1.96 * np.std(savedmeans2)/np.mean(len(np.linspace(.51,2.99,num=10)))
    plt.fill_between(np.linspace(.51,2.99,num=10), (savedmeans2-ci), (savedmeans2+ci), color='b', alpha=.1)
    plt.show()
    