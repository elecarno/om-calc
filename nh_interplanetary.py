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

    def dot_product(self, vec2):
        return (self.x*vec2.x + self.y*vec2.y + self.z*vec2.z)

    def __add__(self, vec2):
        return Vec3D(self.x+vec2.x, self.y+vec2.y, self.z+vec2.z)
    
    def __sub__(self, vec2):
        return Vec3D(self.x-vec2.x, self.y-vec2.y, self.z-vec2.z)

    # def scalar_mult(self, scalar):
    #     return Vec3D(scalar*self.x, scalar*self.y, scalar*self.z)
    
    def __mul__(self, scalar):
        return Vec3D(scalar*self.x, scalar*self.y, scalar*self.z)

    def __rmul__(self, scalar):
        return Vec3D(scalar*self.x, scalar*self.y, scalar*self.z)

    def cross_product(self, vec2):
        i = (self.y*vec2.z - self.z*vec2.y)
        j = -(self.z*vec2.x - self.x*vec2.z)
        k = (self.x*vec2.y - self.y*vec2.x)
        return Vec3D(i, j, k)

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

    def get_position_at_time(self, time):
        T = (2*PI)/(self.grav_p**2)*(self.ang_mom/(math.sqrt(1-self.ecc**2)))**3 # period
        M = ((2*PI)/T)*time # mean anomaly
        E = ecc_anom(M, self.ecc, 15) # eccentric anomaly
        theta = 2*math.atan(math.sqrt( (1+self.ecc)/(1-self.ecc) ) * math.tan(E/2)) # true anomaly
        r = ((self.ang_mom**2)/self.grav_p)*(1/(1+self.ecc * math.cos(theta))) # orbital equation

        r_vec = q_transform(r, theta, self.arg_p, self.ra_an, self.inclination)
        return r_vec

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

sun = Star("82 G. Eridani", 647001000, 1.5900e30)
planets = []

# INTERPLANETARY TRAJECTORY VARIABLES
P_DEPARTURE = 8
P_ARRIVAL = 9
Z_VAL_0 = 39.5
DIRECTION = 1 # 1 for prograde, 0 for retrograde
TRAVEL_TIME = 90 # in days
ALTITUDE_DEP = 5e7 # altitude of circular parking orbit for departure
ALTITUDE_ARR = 5e7 # altitude of periapsis of arrival orbit
ECC_ARR = 0.4 # eccentricity of arrival orbit


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


# ORBITAL TRAJECTORY FUNCTIONS
def x_p(t, r_p, a, b, w, W, i):
    x_p = ( 
        ((r_p - a) + a*np.cos(t))*(np.cos(W)*np.cos(w)-np.sin(W)*np.sin(w)*np.cos(i)) 
        + (b*np.sin(t))*(-np.cos(W)*np.sin(w)-np.sin(W)*np.cos(i)*np.cos(w))
        )
    return x_p

def y_p(t, r_p, a, b, w, W, i):
    y_p = (
        ((r_p - a) + a*np.cos(t))*(np.sin(W)*np.cos(w)+np.cos(W)*np.cos(i)*np.sin(w))
        + (b*np.sin(t))*(-np.sin(W)*np.sin(w)+np.cos(W)*np.cos(i)*np.cos(w))
    )
    return y_p

def z_p(t, r_p, a, b, w, W, i):
    z_p = (
        ((r_p - a) + a*np.cos(t))*np.sin(i)*np.sin(w) 
        + (b*np.sin(t))*np.sin(i)*np.cos(w))
    return z_p


# INTERPLANETARY TRAJECTORY
# Lambert's Problem
grav_p_sun = CONST_G * sun.mass

r_1 = planets[P_DEPARTURE-1].get_position_at_time(t) # position of departure
r_2 = planets[P_ARRIVAL-1].get_position_at_time(t+TRAVEL_TIME*(24*3600)) # position of arrival

# change in true anomaly:
r_cross_vec = r_1.cross_product(r_2)
r_Z = r_cross_vec.z  # (r_1.x*r_2.y - r_1.y*r_2.x)

D_theta = 0
if DIRECTION == 1:
    if r_Z >= 0:
        D_theta = math.acos( (r_1.dot_product(r_2)) / ( r_1.magnitude()*r_2.magnitude()) )
    if r_Z < 0:
        D_theta = 2*PI - math.acos( (r_1.dot_product(r_2)) / ( r_1.magnitude()*r_2.magnitude()) )
elif DIRECTION == 0:
    if r_Z < 0:
        D_theta = math.acos( (r_1.dot_product(r_2)) / ( r_1.magnitude()*r_2.magnitude()) )
    if r_Z >= 0:
        D_theta = 2*PI - math.acos( (r_1.dot_product(r_2)) / ( r_1.magnitude()*r_2.magnitude()) )

# "A" value
A = math.sin(D_theta)*math.sqrt( (r_1.magnitude()*r_2.magnitude()) / (1-math.cos(D_theta)) )

inf_max = 30 # max iterations for infinite series

def S_func(z, max):
    output = 0
    for k in range(0, inf_max):
        output += ( (-1)**k ) * ( (z**k) / math.factorial((2*k + 3)) )
    return output

def C_func(z, max):
    output = 0
    for k in range(0, inf_max):
        output += ( (-1)**k ) * ( (z**k) / math.factorial((2*k + 2)) )
    return output

def y_func(z):
    return r_1.magnitude() + r_2.magnitude() + A * (
        ( z * S_func(z, inf_max) - 1 ) / ( math.sqrt(C_func(z, inf_max)) )
        )

def F_func(z):
    return (
        ( math.pow(( (y_func(z)) / (C_func(z, inf_max)) ), (3/2)) )
        *S_func(z, inf_max) + A*(math.sqrt(y_func(z)))
        - math.sqrt(grav_p_sun)*(TRAVEL_TIME*(24*3600))
    )

def F_der_func(z):
    output = 0

    if abs(z) > 0:
        output = (
            math.pow(( y_func(z) / C_func(z, inf_max) ), (3/2))
            * (
                (1/(2*z))
                * ( C_func(z, inf_max) - (3/2)*( S_func(z, inf_max) / C_func(z, inf_max) ) )
                + (3/4) * ( (S_func(z, inf_max)**2) / C_func(z, inf_max) )
            )
            + (A/8) * ( 3 * ( S_func(z, inf_max) / C_func(z, inf_max) ) * math.sqrt(y_func(z)) )
        )
    elif z == 0:
        output = -(7/240)
    
    return output


def calculate_z_final(z_start, iterations):
    z_current = 0
    z_previous = z_start
    for i in range(0, iterations):
        z_current = z_previous - ( F_func(z_previous) / F_der_func(z_previous))
        z_previous = z_current
    return z_current

z_final = calculate_z_final(39.5, 5)

# z_1 = Z_VAL_0 - F_func(Z_VAL_0)/F_der_func(Z_VAL_0)
# z_2 = z_1 - F_func(z_1)/F_der_func(z_1)
# z_3 = z_2 - F_func(z_2)/F_der_func(z_2)
# z_4 = z_3 - F_func(z_3)/F_der_func(z_3)
# z_final = z_4 - F_func(z_4)/F_der_func(z_4)

# lagrange functions
y_val = y_func(z_final)

lag_f = 1 - y_val/r_1.magnitude()
lag_g = A * math.sqrt(y_val / grav_p_sun)
lag_gt = 1 - y_val/r_2.magnitude()

# velocity vectors
v_1 = (1/lag_g)*(r_2 - lag_f*r_1)
v_2 = (1/lag_g)*(lag_gt*r_2 - r_1)



# trajectory elements
ang_mom_T_vec = r_1.cross_product(v_1)
ang_mom_T = ang_mom_T_vec.magnitude()

inc_T = 2*PI - math.acos( (ang_mom_T_vec.z)/ang_mom_T )

N_vec = Vec3D( (-1*ang_mom_T_vec.y), (-1*ang_mom_T_vec.x), 0 )
N = N_vec.magnitude()

ra_an_T = 0
if N_vec.y >= 0:
    ra_an_T = math.acos(N_vec.x/N)
else:
    ra_an_T = 2*PI - math.acos(N_vec.x/N)

v_1_r = (r_1.dot_product(v_1))/r_1.magnitude() # radial velocity

ecc_T_vec = (1/grav_p_sun)*( (v_1.magnitude()**2 - (grav_p_sun/r_1.magnitude()))*r_1 - (r_1.magnitude()*v_1_r*v_1) )
ecc_T = ecc_T_vec.magnitude()

arg_p_T = 2*PI
if ecc_T_vec.z >= 0:
    arg_p_T -= math.acos( N_vec.dot_product(ecc_T_vec)/(N*ecc_T) )
else:
    arg_p_T -= 2*PI - math.acos( N_vec.dot_product(ecc_T_vec)/(N*ecc_T) )


a_T = ((ang_mom_T**2)/grav_p_sun)*(1/(1 - ecc_T**2))
b_T = a_T*math.sqrt(1 - ecc_T**2)
r_p_T = (ang_mom_T**2/grav_p_sun)*(1/(1 + ecc_T))


# print Velocity Information
a_1 = ((planets[P_DEPARTURE-1].ang_mom**2)/planets[P_DEPARTURE-1].grav_p)*(1/(1-planets[P_DEPARTURE-1].ecc**2))
a_2 = ((planets[P_ARRIVAL-1].ang_mom**2)/planets[P_ARRIVAL-1].grav_p)*(1/(1-planets[P_ARRIVAL-1].ecc**2))

v_he_departure = v_1.magnitude() - math.sqrt( 
    planets[P_DEPARTURE-1].grav_p * ( 2/r_1.magnitude() - 1/a_1) 
    )

v_he_arrival = math.sqrt( 
    planets[P_ARRIVAL-1].grav_p * ( 2/r_2.magnitude() - 1/a_2) 
    ) - v_2.magnitude()

print(f"{planets[P_DEPARTURE-1].name} to {planets[P_ARRIVAL-1].name} in {TRAVEL_TIME} days")
print("-------------------------------------------------------------")

print("Hyperbolic Excess Speeds:")
print(f" - Departure:  {round(v_he_departure)} m/s")
print(f" - Arrival:   {round(v_he_arrival)} m/s")

print("Trajectory Elements:")
print(f" - Angular Mom.:    {round(ang_mom_T)} m^2/s")
print(f" - Eccentricity:    {ecc_T}")
print(f" - Inclination:     {round(inc_T * (180/PI))} deg")
print(f" - RA of Asc. Node: {round(ra_an_T * (180/PI))} deg")
print(f" - Arg. of Peri.:   {round(arg_p_T * (180/PI))} deg")


# DEPARTURE DELTA-V
grav_p_dep = CONST_G * planets[P_DEPARTURE-1].mass # grav p of spacecraft around departure planet.
r_p_dep = planets[P_DEPARTURE-1].radius + ALTITUDE_DEP # periapsis radius of departure, also height of circular parking orbit.

ang_mom_D = r_p_dep * math.sqrt( v_he_departure**2 + (2*grav_p_dep)/(r_p_dep) )
# ecc_D = 1 + (r_p_dep * (v_he_departure**2))/(grav_p_dep)
# a_D = ((ang_mom_D**2)/(grav_p_dep))*(1/(ecc_D**2 - 1))
# b_D = a_D * math.sqrt(ecc_D**2 - 1)
v_p_dep = ang_mom_D/r_p_dep
v_c_dep = math.sqrt( grav_p_dep / r_p_dep )

delta_v_departure = v_p_dep - v_c_dep


# ARRIVAL DELTA-V
grav_p_arr = CONST_G * planets[P_ARRIVAL-1].mass
r_p_arr = planets[P_ARRIVAL-1].radius + ALTITUDE_ARR

v_p_hyperbola_arr = math.sqrt( v_he_arrival**2 + (2*grav_p_arr)/(r_p_arr) )
v_p_capture_arr = math.sqrt( (grav_p_arr*(1+ECC_ARR))/r_p_arr )

delta_v_arrival = v_p_hyperbola_arr - v_p_capture_arr

print("Delta-v:")
print(f" - Departure:  {round(delta_v_departure)} m/s")
print(f" - Arrival:    {round(delta_v_arrival)} m/s")
print(f" - Total:      {round(delta_v_departure + delta_v_arrival)} m/s")


# DRAW PLOT ----------------------------------------------------------------------------------------
x_departure = []
y_departure = []
colours_departure = []
x_arrival = []
y_arrival = []
colours_arrival = []

for p in planets:
    x_departure.append(p.get_position_at_time(t).x / S_FAC)
    y_departure.append(p.get_position_at_time(t).y / S_FAC)
    colours_departure.append("b")

for p in planets:
    x_arrival.append(p.get_position_at_time(t+TRAVEL_TIME*(24*3600)).x / S_FAC)
    y_arrival.append(p.get_position_at_time(t+TRAVEL_TIME*(24*3600)).y / S_FAC)
    colours_arrival.append("g")

# Select length of axes and the space between tick labels
xmin, xmax, ymin, ymax = -60000, 60000, -60000, 60000

# Plot planet points
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(0, 0, c="r", zorder=1)
ax.scatter(x_departure, y_departure, c=colours_departure, zorder=1)
ax.scatter(x_arrival, y_arrival, c=colours_arrival, zorder=1)

# Plot planet trajectories
t_range = np.arange(0, 2*PI, PI/256)

for p in planets:
    a = ((p.ang_mom**2)/p.grav_p)*(1/(1-p.ecc**2)) # semimajor axis
    b = a*math.sqrt(1-p.ecc**2) # semiminor axis
    r_p = ((p.ang_mom**2)/p.grav_p)*(1/(1+p.ecc)) # periapsis radius

    plt.plot(
        x_p(t_range, r_p/S_FAC, a/S_FAC, b/S_FAC, p.arg_p, p.ra_an, p.inclination),
        y_p(t_range, r_p/S_FAC, a/S_FAC, b/S_FAC, p.arg_p, p.ra_an, p.inclination),
        "0.8", zorder=0
        )

# Plot transfer trajectory
t_range_high = np.arange(0, 2*PI, PI/65536)

plt.plot(
    x_p(t_range_high, r_p_T/S_FAC, a_T/S_FAC, b_T/S_FAC, arg_p_T, ra_an_T, inc_T),
    y_p(t_range_high, r_p_T/S_FAC, a_T/S_FAC, b_T/S_FAC, arg_p_T, ra_an_T, inc_T),
    "r", zorder=1
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