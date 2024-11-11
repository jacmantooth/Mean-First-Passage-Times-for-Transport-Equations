import numpy as np 
import math 
import random 
import matplotlib.pyplot as plt 
from scipy.special import i0
'''
The goal of this code is to fully vectorize the process, optimizing computational efficiency to handle millions of test cases 
based on the problem defined in our paper. This code simulates the behavior of a particle within a cell using the velocity jump process, 
where the objective is to reach the boundary. Given the infinite-horizon nature of this problem, our approach ensures a feasible solution.
'''




def distance(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

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
    mainindex = ~(var2var0 & var1var3)
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
  return 1.78*np.ones(len(x))

def q(x,y,phi): # finding our q 
    theta = np.arctan2(y,x)
    return(1/(2*np.pi*i0(k(x,y)))) * np.cosh(k(x,y)*np.sin(phi-theta))

def get_angle(x,y): # for each new position we get the angle that it should go 
    angle = np.zeros(len(x))
    M = np.cosh(k(x,y))/(2*np.pi*i0(k(x,y)))
    num_acc = np.zeros(len(x))
    index = (num_acc == 0)

    while (num_acc == 0).any() :
        u = np.random.rand(len(x))
        angle = 2*np.pi*np.random.rand(len(x))
        qindex = (u < q(x,y,angle)/M )
        num_acc[qindex & index] += angle[qindex& index]
        index = (num_acc == 0)
    return(num_acc) # return the angle 



particule = 1000000 # how many particule we want to find the solution for 
escapetime = np.zeros(particule) #the time at which the paticule reaches the goal 
times = np.zeros(particule)  # the current time 
mu= 1000 #mu 
v = np.sqrt(mu) #nu 
xnew = np.zeros(particule) # our current x position 
ynew = np.zeros(particule) # our current y position
xnew2= np.zeros(particule)# our new x position 
ynew2 = np.zeros(particule) # our new y positon 
radius = np.zeros(particule) # the radius of our goal 
angle = np.zeros(particule) # our angle which we are going 
r = np.zeros(particule) # our current radius aka where we are at 
time_diff = np.zeros(particule) # the time different 
times2 =np.zeros(particule)  # our new time 
xnew[:] = 0 # setting our starting location 
ynew[:] = 0 # setting our starting location 
circler = np.ones(particule) * 3 # def the radius for each one 
savedx = []
savedy = []

inter_time=np.zeros(particule)  
while any(escapetime ==0):
    '''
    we run till all of the escapetime are filled, and only work on the ones that havent found the target yet 
    '''

    zeroidex = (escapetime == 0 ) # we first search for all the ones that havent found the tagert yet 
    

    xwant = xnew[zeroidex] 

    ywant = ynew[zeroidex]

    angle[zeroidex] = get_angle(xwant, ywant) # finding the angle 
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
