#!/usr/bin/env python3

#CRC code 


#import all of the need functions 
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0
from scipy.interpolate import RegularGridInterpolator
import random



def getmindist(cord,xbar):# given a position on the square domain this function finds the closet line to that position 
    min_distance = float('inf')  
    min_index = -1 
    for x in range(len(cord)): #given a vector of positions, this will forloop through all the lines 
        current_distance = distance_point_to_line_segment(xbar,cord[x][:2],cord[x][2:])
        if current_distance < min_distance:
            min_distance = current_distance  # Update the minimum distance
            mind = distance_point_to_line_segment(xbar,cord[x][:2],cord[x][2:])
            m = mind
            
            theta = math.atan(m)   
            min_index = x
    return(mind,theta)#returns how close the particle is to the line and theta 

def distance_point_to_line_segment(point, line_start, line_end):# code to find which point a particule is close to  either end A,B or dig to it 
    line_start_to_point = point - line_start 
    line = line_end - line_start
    # This finds the closest point on the infinite line, but it might not be on the line segment
    line_length_squared = np.dot(line, line)
    projected_length = np.dot(line_start_to_point, line) / line_length_squared
    # Clamp the projected length to the range [0, 1] to ensure the closest point is on the line segment
    projected_length = max(0, min(1, projected_length))
    # Find the closest point on the line segment
    closest_point = line_start + projected_length * line
    # Distance from the point to the closest point on the line segment
    distance = np.linalg.norm(point - closest_point)
  
    return distance #return the distance 



def get_angdist(): #creating a mesh grid. 
    
    cord = np.array([[191,11, 241,234],[241,234,278,452],[278,452, 309,639],[309,639,310,627],[309,627,331,756],[16,237, 279,233],
                    [279,233, 1145,249],[1145,249, 1406,252],[1406,252, 1469,256],[389,10,296,300],[296,300,270,411],
                    [270,411, 149,780],[149,780, 98,950],[98,950,22,1140],[11,455, 170,453],[170,453, 179,447],[179,447, 280,451],
                    [15,360, 222,550],[222,550, 310,639],[310,639, 443,745],[878,10, 746,717],[746,717, 724,828],[724,828, 700,911],
                    [949,6,930,48],[930,48,874,181],[855,119, 880,243],[880,243, 912,396],[912,396, 1003,985],[1003,985,1026,1152],
                    [1026,1152, 1043,1309],[1043,1309, 1050,1467],[519,7, 547,116],[547,116,628,560],[628,560,721,1002],[721,1002,744,1141],
                    [744,1141,788,1326],[788,1326,812,1465],[24,795, 330,753],[330,753, 447,747],[447,747, 962,698],[962,698,1184,701],
                    [1184,701,1220,698],[1220,698,1465,667],[1275,6,1291,26],[1291,26,1318,47],[1318,47,1350,69],[1350,69,1390,82],
                    [1390,82,1423,344],[1423,344,1469,666],[923,460,1049,572],[1049,572,1180,702],[1118,84, 1195,534],[1195,534,1279,1134],
                    [1279,1134,1323,1465],[929,12,1219,616],[1219,616,1317,809],[1317,809,1379,958],[1379,958,1463,1134],[450,750,618,930],
                    [618,930,723,1031],[723,1031,837,1149],[837,1149,1001,1314],[843,1470, 907,1401],[907,1401,1270,1065],[1270,1065,1445,888],
                    [1445,888,1468,843],[996,1318,1138,1467],[13,1117,266,1119],[266,1119,369,1121],[369,1121,487,1124],[487,1124,680,1140],
                    [680,1140,831,1148],[831,1148,1184,1147],[1184,1147,1368,1124],[1368,1124,1465,1121],[1287,1203,1345,1243],[1345,1243,1469,1329],
                    [13,1117, 105,1179],[105,1179,452,1386],[485,1315,587,1387],[587,1387,723,1470],[15,1337,327,1316],[327,1316,490,1315],
                    [490,1315,789,1328],[789,1328,978,1334],[978,1334,1071,1335],[1071,1335,1096,1330],[1096,1330,1469,1329],[421,1122,437,1222],
                    [437,1222,463,1468],[98,950,124,980],[124,980,160,995],[160,995, 300,1000],[300,1000,359,1017],[359,1017,378,1038],
                    [378,1038,395,1078],[395,1078,487,1124]
    ])
    #why we do this? bc other wise the search for the closest line would take forever. So we create a mesh and at every point in that mesh we find 
    #which line is closet to that point. this allows us to est which line is closest without going through the def function above saving us time
    nxc, nyc = 1200,1200;
    xc= np.linspace(0,1470,nxc); yc = np.linspace(0,1479,nyc);
    [xcv,ycv] = np.meshgrid(xc,yc,indexing='ij');
    distc = np.zeros((nxc,nyc));
    dista = np.zeros((nxc,nyc));
    for i in range(nxc):
        for j in range(nyc):
            d1,ang = getmindist(cord,[xcv[i,j],ycv[i,j]])
            distc[i,j] = d1;
            dista[i,j] = ang;
    interp = RegularGridInterpolator((xc, yc), distc)
    anginterp = RegularGridInterpolator((xc, yc), dista)
    return(interp, anginterp)

def k(x,y): #finding k, k tells how strongly align we are with the line, and angle 

    xbar = np.dstack((x, y))[0]
    
    smallv = interp(xbar)
  
    angle= anginterp(xbar)
    indices = np.where(smallv<10) #if the particule is less then 10 units away we randomly diffuse other wise we are align with the closest line
    k = np.zeros(len(smallv))
    k[indices] = 25
    
    return (k,angle) 

def get_angle(kt,angle):
    #using the Von Mises distribution we get our angle

    # we use a know distribution 
    placeholderangle = 2*np.pi*np.random.rand(len(kt))
    M = np.cosh(kt)/(2*np.pi*i0(kt))
    u = np.random.rand(len(kt))
    placeholder =  q(placeholderangle,kt,angle)/M
    indices = np.where((u > placeholder))
    while (u > placeholder).any():
        placeholderangle[indices[0]] = 2 * np.pi * np.random.rand(len(indices[0]))
        placeholder[indices[0]] =  q(placeholderangle[indices[0]],kt[indices[0]],angle[indices[0]])/M[indices[0]]         
        indices = np.where(u > placeholder)       
    
    return(placeholderangle)

def q(phi,k,angle_in_degrees):
    # here we get our q 
    #note if alpha is less 0 it is sin 
    return (1/(2*np.pi*i0(k))) * np.cosh(k*np.cos(phi-angle_in_degrees))

#finding where the a particule cross the boundary 
def findxy(var0,var1,var2,var3):
    x= 0 
    y=0
    if var2<0:
        m = (var3-var1)/(var2-var0)#getting m 
        b = var3- m*var2 # getting b
        x = 0
        y = m*0 + b 
        
    elif var2 > 1470:
        m = (var3-var1)/(var2-var0)#getting m 
        b = var3- m*var2 # getting b
        x = 1470
        y = m*1470 + b

    elif var3 < 0:
        m = (var3-var1)/(var2-var0)#getting m 
        b = var3- m*var2 # getting b
        y= 0 
        x = -b/m
        
    elif var3 > 1479:
        m = (var3-var1)/(var2-var0)#getting m 
        b = var3- m*var2 # getting b
        y = 1479
        x = (y-b)/m
    return(x,y)

#new way
def findxy2(var0,var1,var2,var3):#finding where the a particule cross the boundary 
    
    x= np.zeros(len(var3))
    y=np.zeros(len(var3))
    if any(var2<0):
        indices = np.where((var2 < 0))
        m = (var3[indices]-var1[indices])/(var2[indices]-var0[indices])#getting m 
        b = var3[indices]- m*var2[indices] # getting b
        x[indices] = 0
        y[indices] = m*0 + b 
     
    if any(var2 > 1470):
        indices = np.where((var2 > 1470))
        m = (var3[indices]-var1[indices])/(var2[indices]-var0[indices])#getting m 
        b = var3[indices]- m*var2[indices] # getting b
        x[indices] = 1470
        y[indices]= m*1470 + b

    if any(var3 < 0):
       
        indices = np.where((var3 < 0))
        m = (var3[indices]-var1[indices])/(var2[indices]-var0[indices])#getting m 
        b = var3[indices]- m*var2 [indices]# getting b
        y[indices]= 0 
        x[indices] = -b/m
        
    if any(var3 > 1479):
        indices = np.where((var3 > 1479))
        m = (var3[indices]-var1[indices])/(var2[indices]-var0[indices])#getting m 
        b = var3[indices]- m*var2[indices] # getting b
        y[indices] = 1479
        x[indices] = (y[indices]-b)/m
    return(x,y)


if __name__ == "__main__":
    #start by creating the mesh and getting a mesh of dist and angle 
    interp, anginterp = get_angdist()


    mu= 1e4 
    v = 100*np.sqrt(mu)#speed on MT
    rounds = 2#how many eps we are running
    #random starting point 


    # time it took to escape 
     # rand picking a angle 

    


    #setting everything to zero 
    xnew = np.zeros(rounds)
    ynew = np.zeros(rounds)


    xtestnew = np.zeros(rounds)
    ytestnew = np.zeros(rounds)

    escape_time = np.zeros(rounds)
    amout_capt = np.zeros(rounds)
    times = np.zeros(rounds)
    
    numloops = 0 
    # choose a starting point
    startingxloc = 100
    startingyloc = 100


    for n in range(rounds):
       
        #xnew[n] +=random.randint(0, 1479)
        #ynew[n] +=random.randint(0, 1470)
        xnew[n] +=startingxloc
        ynew[n] +=startingyloc
    #adds a row so we can add our new location to it
    xtestnew = np.vstack([xtestnew, np.zeros(rounds)])
    ytestnew = np.vstack([ytestnew, np.zeros(rounds)])
    xnew = np.vstack([xnew, np.zeros(rounds)])
    ynew = np.vstack([ynew, np.zeros(rounds)])
    times = np.vstack([times, np.zeros(rounds)])

    n= 0 

   
    
    r = np.zeros(rounds)
    r[:] = random.random()   
    angle  = np.random.rand(rounds)* 2 * math.pi #gets a random angle to start out with

    time_diff = -1/mu *np.log(r)
    #begin to change the different from different time to a same time system:
    #we want this to be on the same clock some we can find where the density is 

    #jump in time and get the new position 
    times[n+1]= times[n]+time_diff
    xnew[n+1] =xnew[n]+v*np.cos(angle)*(time_diff)
    ynew[n+1] =ynew[n]+v*np.sin(angle)*(time_diff)

    #if it goes out we want to find where and when it went outside 
    for ns in np.where((ynew[n+1] < 0) | (ynew[n+1] > 1479) | (xnew[n+1] < 0) |(xnew[n+1] > 1470))[0]:
        indices = np.where((ynew[n+1] < 0) | (ynew[n+1] > 1479) | (xnew[n+1] < 0) |(xnew[n+1] > 1470))
        xnew[n+1][ns],ynew[n+1][ns] = findxy(xnew[n][ns],ynew[n][ns],xnew[n+1][ns],ynew[n+1][ns])
        inter_time = (xnew[n+1][ns]-xnew[n][ns])/(v*np.cos(angle[ns]))
        escape_time[ns] += times[n][ns]+np.abs(inter_time)
        xnew = np.delete(xnew,indices,1)
        ynew = np.delete(ynew,indices,1)
        times = np.delete(times,indices,1)
        #delete the old found index to save comp time 
            
    #create a new row for non escape particule 
    xnew= np.vstack([xnew,  np.zeros(len(xnew[n+1]))])
    ynew= np.vstack([ynew,  np.zeros(len(ynew[n+1]))])
    times = np.vstack([times, np.zeros(len(times[n+1]))])

    xtestnew = np.vstack([xtestnew, np.zeros(rounds)])
    ytestnew = np.vstack([ytestnew, np.zeros(rounds)])

    n=1
    numloops=0
        

    while any(escape_time==0):
        
        kfun,angle_in_degrees = k(xnew[n],ynew[n]) # give a array of position we want the dist and ang 
            

        angle = get_angle(kfun,angle_in_degrees) # now we get the angle that we are heading 
        # time jump based off the angle found 
        r = np.zeros(len(xnew[0]))
        r[:] = random.random()   
        
        time_diff = -1/mu *np.log(r)
        times[n+1] = times[n]+time_diff
        xnew[n+1] =xnew[n]+v*np.cos(angle)*(time_diff)
        ynew[n+1] =ynew[n]+v*np.sin(angle)*(time_diff)
        
        #Looks for any particule that have been captured and find the time it did 
        if ((ynew[n+1] < 0) | (ynew[n+1] > 1479) | (xnew[n+1] < 0) | (xnew[n+1] > 1470)).any():
            indices = np.where((ynew[n+1] < 0) | (ynew[n+1] > 1479) | (xnew[n+1] < 0) |(xnew[n+1] > 1470))
            xnew[n+1][indices],ynew[n+1][indices] = findxy2(xnew[n][indices],ynew[n][indices],xnew[n+1][indices],ynew[n+1][indices])
            
        
            inter_time = (xnew[n+1][indices]-xnew[n][indices])/(v*np.cos(angle[indices]))
            
            escape = times[n][indices]+np.abs(inter_time)
            
            for count in escape:
                escape_time[numloops] += count
                
                numloops+=1
            xtestnew[:,numloops-1]= xnew[:,indices[0][0]][:]
            ytestnew[:,numloops-1] = ynew[:,indices[0][0]][:]

            xnew = np.delete(xnew,indices,1)
            ynew = np.delete(ynew,indices,1)
            times = np.delete(times,indices,1)
        
        
        xnew= np.vstack([xnew,  np.zeros(len(xnew[0]))])
        ynew= np.vstack([ynew,  np.zeros(len(ynew[0]))])
        times = np.vstack([times, np.zeros(len(times[0]))])

        xtestnew = np.vstack([xtestnew, np.zeros(rounds)])
        ytestnew = np.vstack([ytestnew, np.zeros(rounds)])
        n+=1
        
    numc=0
    #orgs the time that they were captured 
    timelistfinal = sorted(escape_time)
    for x in timelistfinal:
        amout_capt[numc:] += 1 
        numc+=1 

    plt.figure()
    plt.plot(timelistfinal,amout_capt/rounds)
    plt.title(f"Location: {startingxloc}x and {startingyloc}y for {numloops} eps with mean: {np.mean(timelistfinal)}" )
    #plt.savefig(f'/afs/crc.nd.edu/user/j/jmantoot/Private/amout_capt_roundsfinal{rounds}.png')
    plt.show()
  
    
    plt.figure()
    plt.title(f"Location: {startingxloc}x and {startingyloc}y for {numloops} eps with mean: {np.mean(timelistfinal)}" )
    plt.hist(timelistfinal)
    #plt.savefig(f'/afs/crc.nd.edu/user/j/jmantoot/Private/histgramogwolftrackfinal{rounds}.png')
    plt.show()
    
 
    indice = np.where(xtestnew[:][:,0:1]==0)[0][0] and np.where(ytestnew[:][:,0:1]==0)[0][0]
    xconcatenated_array2 = xtestnew[:indice][:,0:1]
    yconcatenated_array2 = ytestnew[:indice][:,0:1]
    for n in range(np.shape(xtestnew)[1]-1):
        indice =np.where(xtestnew[:][:,n+1:n+2]==0)[0][0] and np.where(ytestnew[:][:,n+1:n+2]==0)[0][0]
        xconcatenated_array = np.concatenate((xconcatenated_array2, xtestnew[:indice][:,n+1:n+2]))
        yconcatenated_array = np.concatenate((yconcatenated_array2, ytestnew[:indice][:,n+1:n+2]))

    plt.figure()
    plt.hexbin( xconcatenated_array ,  yconcatenated_array ,  cmap='viridis')
    plt.gca().invert_yaxis()
    #plt.savefig(f'/afs/crc.nd.edu/user/j/jmantoot/Private/hexbinfor{rounds}.png')
    plt.show()
    data = np.concatenate((xconcatenated_array,yconcatenated_array),axis=1)
    meshsize = 1500
    xmesh = np.linspace(0,1470,meshsize)
    ymesh = np.linspace(0,1479,meshsize)

    H, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=meshsize)
    H_normalized = H / np.max(H)


    plt.figure()
    plt.imshow(H_normalized.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='viridis')
    plt.colorbar(label='Count')
    plt.gca().invert_yaxis()

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    #plt.savefig(f'/afs/crc.nd.edu/user/j/jmantoot/Private/heatmapfor{rounds}.png')
    plt.show()





