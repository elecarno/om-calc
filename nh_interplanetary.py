import csv
import math
import numpy as np
import matplotlib.pyplot as plt

# UTILITIES ----------------------------------------------------------------------------------------
def rad2deg(rad):
    return rad*(180/math.pi)


# CLASSES ------------------------------------------------------------------------------------------
class Vec3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def  __str__(self):
        return f"({self.x}, {self.y}, {self.z})"


class Star:
    def __init__(self, name, radius, mass):
        self.name = name
        self.radius = radius # meters
        self.mass = mass     # kg

    def __str__(self):
        pr_str = f"\nSTAR"
        pr_str += f"\nName:     \t {self.name}"
        pr_str += f"\nRadius:   \t {self.radius} m"
        pr_str += f"\nMass:     \t {self.mass} kg"

        return pr_str


class Planet:
    def __init__(self, index, name, radius, mass, gravity, ang_mom, inc, ra_an, arp_p, ecc, grav_p):
        # main attributes
        self.index = index
        self.name = name
        self.radius = radius     # meters
        self.mass = mass         # kilograms
        self.gravity = gravity   # newtons

        # orbital elements
        self.ang_mom = ang_mom   # m^2/s,            Specific Angular Momentum
        self.inclination = inc   # radians,          Inclination
        self.ra_an = ra_an       # radians,          RA of Ascending Node
        self.arg_p = arp_p       # radians,          Argument of Periapsis
        self.ecc = ecc           # dimensionless,    Eccentricity
        self.t_anom = 0          # radians,          True Anomaly,

        # gravitational parameter
        self.grav_p = grav_p

        # positions
        self.pos = Vec3D(0, 0, 0)

    def __str__(self):
        pr_str = f"\nPLANET {self.index}"
        pr_str += f"\nName:     \t {self.name}"
        pr_str += f"\nRadius:   \t {self.radius} m"
        pr_str += f"\nMass:     \t {self.mass} kg"
        pr_str += f"\nGravity:  \t {self.gravity} N ({round(self.gravity/9.81, 1)} g)"

        pr_str += f"\nAngular Momentum: \t{self.ang_mom} m^2/s"
        pr_str += f"\nEccentricity:     \t{self.ecc}"
        pr_str += f"\nInclination:      \t{self.inclination} rad ({round(rad2deg(self.inclination),2)} deg)"
        pr_str += f"\nRA of Node:       \t{self.ra_an} rad ({round(rad2deg(self.ra_an),2)} deg)"
        pr_str += f"\nArg. of Periapsis:\t{self.arg_p} rad ({round(rad2deg(self.arg_p),2)} deg)"
        pr_str += f"\nTrue Anomaly:     \t{self.t_anom} rad ({round(rad2deg(self.t_anom),2)} deg)"
        
        pr_str += f"\nGravitational Para.: \t{self.grav_p} m^3/s^2"

        return pr_str


# GLOBALS ------------------------------------------------------------------------------------------
CONST_G = 6.6742e-11 # Gravitational constant
PI = math.pi
S_FAC = 2.0e8 # scaling factor for plotting

t = 746000000 # global time variable (seconds)

sun = Star("82 G. Eridani", 647001000, 1.59e30)
planets = []


# PROGRAM ------------------------------------------------------------------------------------------
# Initialise
with open('eri_planets.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader, None)

    # read csv data and create planet objects 
    idx = 1
    for row in csv_reader:
        grav_p = CONST_G*(sun.mass + float(row[3]))
        planet = Planet(idx, row[0], float(row[1])*1000, float(row[3]), float(row[5]), float(row[19]), float(row[8]), float(row[10]), float(row[12]), float(row[13]), grav_p)
        planets.append(planet)
        idx += 1


# function for infinite series calculation of Eccentric Anomaly
def ecc_anom(M, e, max): # max is the number of terms to sum
    # define E with initial term of Mean Anomaly
    E = M
    
    # infinite series
    for n in range(1, max):
        # bessel function
        J = 0
        for k in range(0, max-1):
            J += ( (-1)**k) / (math.factorial(k)*math.factorial((n+k)) )*((n*e)/2)**(n+2*k)
        
        # add terms to E using bessel function
        E += (2/n) * J * math.sin(n*M)

    return E

# vector transformation from orbital plane to geocentric equatorial frame
def q_transform(factor, th, w, W, i):
    Q = Vec3D(0, 0, 0)
    
    Q.x = factor * ( 
        math.cos(th)*(math.cos(W)*math.cos(w)-math.sin(W)*math.sin(w)*math.cos(i)) 
        + math.sin(th)*(-math.cos(W)*math.sin(w)-math.sin(W)*math.cos(i)*math.cos(w))
        )

    Q.y = factor * (
        math.cos(th)*(math.sin(W)*math.cos(w)+math.cos(W)*math.cos(i)*math.sin(w))
        + math.sin(th)*(-math.sin(W)*math.sin(w)+math.cos(W)*math.cos(i)*math.cos(w))
    )

    Q.z = factor * (math.cos(th)*math.sin(i)*math.sin(w) + math.sin(th)*math.sin(i)*math.cos(w))

    return Q


# Calculate positions of planets
for p in planets:
    T = (2*PI)/(p.grav_p**2)*(p.ang_mom/(math.sqrt(1-p.ecc**2)))**3 # period
    M = ((2*PI)/T)*t # mean anomaly
    E = ecc_anom(M, p.ecc, 15) # eccentric anomaly
    theta = 2*math.atan(math.sqrt( (1+p.ecc)/(1-p.ecc) ) * math.tan(E/2)) # true anomaly
    r = ((p.ang_mom**2)/p.grav_p)*(1/(1+p.ecc * math.cos(theta))) # orbital equation

    r_vec = q_transform(r, theta, p.arg_p, p.ra_an, p.inclination)
    p.pos = r_vec


# ORBITAL TRAJECTORY FUNCTIONS
def x_p(t, r_p, a, b, w, W, i):
    x_p = ( 
        ((r_p - a) + a*np.cos(t))*(np.cos(W)*np.cos(w)-np.sin(W)*np.sin(w)*np.cos(i)) 
        + (b*np.sin(t))*(-np.cos(W)*np.sin(w)-np.sin(W)*np.cos(i)*np.cos(w))
        )
    print(x_p)
    return x_p

def y_p(t, r_p, a, b, w, W, i):
    y_p = (
        ((r_p - a) + a*np.cos(t))*(np.sin(W)*np.cos(w)+np.cos(W)*np.cos(i)*np.sin(w))
        + (b*np.sin(t))*(-np.sin(W)*np.sin(w)+np.cos(W)*np.cos(i)*np.cos(w))
    )
    print(y_p)
    return y_p

def z_p(t, r_p, a, b, w, W, i):
    z_p = (
        ((r_p - a) + a*np.cos(t))*np.sin(i)*np.sin(w) 
        + (b*np.sin(t))*np.sin(i)*np.cos(w))
    print(z_p)
    return z_p

t_range = np.arange(0, 2*PI, PI/256)


# DRAW PLOT ----------------------------------------------------------------------------------------
x = [0]
y = [0,]

colours = ['r']

for p in planets:
    x.append(p.pos.x / S_FAC)
    y.append(p.pos.y / S_FAC)
    colours.append("b")

# Select length of axes and the space between tick labels
xmin, xmax, ymin, ymax = -60000, 60000, -60000, 60000
ticks_frequency = 5000

# Plot points
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x, y, c=colours)

# Plot trajectories
for p in planets:
    a = ((p.ang_mom**2)/p.grav_p)*(1/(1-p.ecc**2)) # semimajor axis
    b = a*math.sqrt(1-p.ecc**2) # semiminor axis
    r_p = ((p.ang_mom**2)/p.grav_p)*(1/(1+p.ecc)) # periapsis radius

    plt.plot(
        x_p(t_range, r_p/S_FAC, a/S_FAC, b/S_FAC, p.arg_p, p.ra_an, p.inclination),
        y_p(t_range, r_p/S_FAC, a/S_FAC, b/S_FAC, p.arg_p, p.ra_an, p.inclination)
        )

# Set identical scales for both axes
ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')

# Set bottom and left spines as x and y axes of coordinate system
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()