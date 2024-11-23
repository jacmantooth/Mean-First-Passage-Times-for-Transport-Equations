import numpy as np 
import matplotlib.pyplot as plt 

"""

Building the code for two circle of radius r and R with lines that connect them to each other 

"""

def connect_points(bigradius,smallradius,numlines):
    '''
    Input:three input big radius small radius and num of lines
    Outout: the coord that connect each point 
    '''

    angles = np.linspace(0, 2 * np.pi, numlines, endpoint=False)

    small__circlepoint = np.array([
    [smallradius * np.cos(angle), smallradius * np.sin(angle)] for angle in angles
    ])

    large_circle_points = np.array([
    [bigradius * np.cos(angle), bigradius * np.sin(angle)] for angle in angles
    ])

    return (small__circlepoint,large_circle_points)

def graph(bigradius,smallradius,small__circlepoint,large_circle_points,numlines):
    theta = np.linspace(0, 2 * np.pi, 500)
    plt.figure(figsize=(8, 8))
    plt.plot(bigradius * np.cos(theta), bigradius * np.sin(theta), label=f"Large Circle (radius={bigradius})")
    plt.plot(smallradius * np.cos(theta), smallradius * np.sin(theta), label=f"Small Circle (radius={smallradius})")

    # Plot the lines connecting the circles
    for i in range(numlines):
        plt.plot(
            [small__circlepoint[i, 0], large_circle_points[i, 0]],
            [small__circlepoint[i, 1], large_circle_points[i, 1]],
            'r-'
        )
    plt.scatter(small__circlepoint[:, 0], small__circlepoint[:, 1], color='blue', label="Small Circle Points")
    plt.scatter(large_circle_points[:, 0], large_circle_points[:, 1], color='green', label="Large Circle Points")

    # Formatting
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.title(f"{numlines} Lines Connecting Two Circles")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid()
    plt.show()

bradius = 5
sradius = 1 
numlines =  5 

spoint,bpoints = connect_points(bradius,sradius,numlines)

graph(bradius,sradius,spoint,bpoints,numlines)


print(spoint,bpoints)