import numpy as np 
import math 
import random 
import matplotlib.pyplot as plt 
from scipy.special import i0
from scipy.interpolate import RegularGridInterpolator
"""
The objective of this code was to verify the results presented 
in Section 6.1.1, Exit from a Disk,
in our paper on Mean First Passage Time. The goal is to analyze
various points within the disk and ensure their behavior aligns 
with the theoretical predictions.
"""


def get_angdist():
    """
    Input: None

    Output: The result for every mesh point allowing us to call 
    it later on for fast comp speed 
    """
    nxc, nyc = 800,800;
    xc= np.linspace(-4,4,nxc); yc = np.linspace(-4,4,nyc);
    [xcv,ycv] = np.meshgrid(xc,yc,indexing='ij');
    distc = np.zeros((nxc,nyc));
    dista = np.arctan2(ycv, xcv)  
   
    interp = RegularGridInterpolator((xc, yc), distc)
    anginterp = RegularGridInterpolator((xc, yc), dista)
    return(interp, anginterp)



def findxy(var0,var1,var2,var3,r):
    """
    Input: Four vectors representing the positions of each particle before and after the velocity jump.

    Output: The intersection point(s) of each particle's path with a circular boundary. 
    We re only interested in where each particle first intersects with the circle, as they are considered "captured" once they reach this boundary. 
    We do not calculate their final positions post-intersection.
    """
    x = np.zeros(len(var0)) # creating two empty vector for our solution that
    y = np.zeros(len(var0))

    # The First case we must deal with is what happen we our final and begin position are on top of each other
    epsilon = 1e-5  # or choose an appropriate tolerance level
    var2var0 = (abs(var2 - var0) < epsilon)
    y[var2var0] = np.sqrt(r[var2var0]**2-var0[var2var0]**2)
    x[var2var0] = var0[var2var0]
    y[var2var0 & (var1<0)]*= -1
    # The next case is what happens when the particule final and begin position are on the same hoz line 
    var1var3 = (abs(var3- var1) < epsilon)
    y[var1var3] = var1[var1var3]
    x[var1var3] =  np.sqrt(r[var1var3]**2 - var1[var1var3]**2)
    x[var1var3 & (var0<0)]*= -1
    
    #now that we have factored the two case we can find the "exact" solution using the simply trig math 
    var2var0 = (abs(var2 - var0) < epsilon)
    var1var3 = (abs(var3- var1) < epsilon)
    mainindex = ~(var2var0 | var1var3)
    m = (var3[mainindex]-var1[mainindex])/(var2[mainindex]-var0[mainindex])#getting m 
    b = var3[mainindex] - m*var2[mainindex] # getting b        
    a1 = (1+m**2)# getting a for x 
    b1 = (2*m*b )#getting b for x
    c1 = (b**2-r[mainindex]**2)#getting c for x

    ay = (1/m**2 +1) #getting a for y 
    by = -((2*b)/m**2)#getting b for y 
    cy = (b**2/m**2 - r[mainindex]**2)#getting c fo y
    
    y1 = (-by+np.sqrt(by**2-4*ay*cy))/(2*ay) 
    y2 = (-by-np.sqrt(by**2-4*ay*cy))/(2*ay)
    x1 = (-b1+np.sqrt(b1**2-4*a1*c1))/(2*a1)         
    x2 = (-b1-np.sqrt(b1**2-4*a1*c1))/(2*a1)

    # now we are return our solution given it satifies the codition if not we flip them, this cause when we interct a circle we can interct two points so this part fixes this issue 
    x[mainindex] = np.where((var0[mainindex] <= x1) & (x1 <= var2[mainindex]), x1, x2)
    y[mainindex] = np.where((var1[mainindex] <= y1) & (y1 <= var3[mainindex]), y1, y2)

    return x,y

def k(x,y):# all k will be the same in this case 
    xbar = np.dstack((x, y))[0]
    
    smallv = interp(xbar)
  
    angle_in_degrees = anginterp(xbar)

    k = np.ones(len(smallv))* 0

    return (k,angle_in_degrees) 

def q(phi,k,angle_in_degrees):
    return (1/(2*np.pi*i0(k))) * np.cosh(k*np.cos(phi-angle_in_degrees))

def get_angle(kfun,angle_in_degrees): # for each new position we get the angle that it should go 
    angle = np.zeros(len(kfun))
    M = np.cosh(kfun)/(2*np.pi*i0(kfun))
    num_acc = np.zeros(len(kfun))
    index = (num_acc == 0)

    while (num_acc == 0).any() :
        u = np.random.rand(len(kfun))
        angle = 2*np.pi*np.random.rand(len(kfun))
        qindex = (u < q(angle,kfun,angle_in_degrees)/M )
        num_acc[qindex & index] = angle[qindex& index]
        index = (num_acc == 0)
    return(num_acc) # return the angle 

def beta(alpha):
    return((2*alpha)/(1+alpha))

interp, anginterp = get_angdist()


xglobal = np.linspace(-2,2,5)
meantimeglobal = np.zeros(xglobal.size)
roundsplace = 0 
for xlocal in xglobal:
    particule = 100000 # how many particule we want to find the solution for 
    xnew = np.zeros(particule) # our current x position 
    ynew = np.zeros(particule) # our current y position
    xnew2= np.zeros(particule)# our new x position 
    ynew2 = np.zeros(particule) # our new y positon 
    xnew[:] = xlocal 
    ynew[:] = 0 


    escapetime = np.zeros(particule) #the time at which the paticule reaches the goal 
    times = np.zeros(particule)  # the current time 
    mu= 2000000  # k equal 1 at 
    v = 40000 #speed on MT
    radius = np.zeros(particule) # the radius of our goal 
    angle = np.zeros(particule) # our angle which we are going 
    r = np.zeros(particule) # our current radius aka where we are at 
    time_diff = np.zeros(particule) # the time different 
    times2 =np.zeros(particule)  # our new time 
    
    outrad = 3 
    circler = np.ones(particule) * outrad# def the radius for each one 
    kfun = np.zeros(particule)  
    angle_in_degrees = np.zeros(particule)  
    inter_time=np.zeros(particule)  
    while any(escapetime ==0):
        '''
        we run till all of the escapetime are filled, and only work on the ones that havent found the target yet 
        '''


        zeroidex = (escapetime == 0 ) # we first search for all the ones that havent found the tagert yet 
        
        kfun[zeroidex] ,angle_in_degrees[zeroidex]  = k(xnew[zeroidex],ynew[zeroidex])
        angle = get_angle(kfun,angle_in_degrees)
        xwant = xnew[zeroidex] 

        ywant = ynew[zeroidex]

        
        r[zeroidex] = np.random.rand(len(xwant)) # find the random r for each one 
    

        time_diff[zeroidex]  = -1/mu *np.log(r[zeroidex]) # getting our time jump
        times2[zeroidex]=times[zeroidex]+time_diff[zeroidex] # add the time jump to our current time 
        
        

        
        xnew2[zeroidex] =xnew[zeroidex]+v*np.cos(angle[zeroidex])*time_diff[zeroidex] # getting the new position for both x and y 
        ynew2[zeroidex] =ynew[zeroidex]+v*np.sin(angle[zeroidex])*time_diff[zeroidex]
        radius[zeroidex] = np.sqrt( xnew2[zeroidex]**2 + ynew2[zeroidex]**2 )  # find the location of both 
        findindex = (radius >= 3) & (escapetime == 0) # seeing which ones are both out side/ at the target and has it escaped already  
        
        
        if findindex.any():
            # finding at what time did it escape if it jumped past our target radius 
            xnew2[findindex],ynew2[findindex]= findxy(xnew[findindex],ynew[findindex],xnew2[findindex],ynew2[findindex],circler[findindex])
            inter_time[findindex] = (xnew2[findindex]-xnew[findindex])/(v*np.cos(angle[findindex])) 
            escapetime[findindex] = times[findindex]+np.abs(inter_time[findindex])# the time at which it escaped 
            
        
        xnew[zeroidex] = xnew2[zeroidex] # adding our new postion and time as our current position and time 
        ynew[zeroidex]  =ynew2[zeroidex] 
        times[zeroidex]  = times2[zeroidex] 



    meantimeglobal[roundsplace] = np.mean(escapetime)
    print(f"mfpt: {np.mean(escapetime)}")
    roundsplace +=1  
    ''' 
    amout_capt = np.zeros(particule)
    timelistfinal = sorted(escapetime)
    numc=0
    for x in timelistfinal:
            amout_capt[numc:] += 1 
            numc+=1 

    print(f"mfpt: {np.mean(timelistfinal)}")
    plt.plot(timelistfinal,amout_capt/particule)
    plt.title(f"mfpt: {np.mean(timelistfinal)}")
    plt.show()
    '''
yglobal = np.zeros(xglobal.size)
radius = np.sqrt(xglobal**2 + yglobal**2)
D = v**2/((2*mu))
sol = 1/(4*D) * (outrad**2 - radius**2)


print(f"The error found was {(sol - meantimeglobal)/sol*100}")

plt.plot(xglobal, sol, label="True Solution")
plt.plot(xglobal, meantimeglobal, label="Approximation")

plt.xlabel("x")  # Add label for x-axis
plt.ylabel("y")  # Add label for y-axis
plt.legend()     # Add legend to show the labels
plt.title("Comparison of True Solution and Approximation")  # Optional: Add a title
plt.show()       # Display the plot

