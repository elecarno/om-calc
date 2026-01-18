import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter



# CONSTANTS ----------------------------------------------------------------------------------------
G = 6.67430e-11      # m^3/kg/s^2
PI = math.pi
S_FAC = 1e8          # scale for plotting



# UTILITIES ----------------------------------------------------------------------------------------
def rad2deg(rad):
    return rad*(180/math.pi)

def scale_formatter(x, pos):
    return f"{x * S_FAC:.0f}"

def scale_formatter_km(x, pos):
    return f"{x * S_FAC / 1e3 :.0f}"

def scale_formatter_sci(x, pos):
    return f"{x * S_FAC :.2e}"

def scale_formatter_sci_km(x, pos):
    return f"{x * S_FAC / 1e3 :.1e}"



# CLASSES ------------------------------------------------------------------------------------------
class Vec3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def mag(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def dot(self, other):
        return (self.x*other.x + self.y*other.y + self.z*other.z)

    def cross(self, other):
        i = (self.y*other.z - self.z*other.y)
        j = -(self.z*other.x - self.x*other.z)
        k = (self.x*other.y - self.y*other.x)
        return Vec3D(i, j, k)

    def __add__(self, other):
        return Vec3D(self.x+other.x, self.y+other.y, self.z+other.z)

    def __sub__(self, other):
        return Vec3D(self.x-other.x, self.y-other.y, self.z-other.z)

    def __mul__(self, scalar):
        return Vec3D(scalar*self.x, scalar*self.y, scalar*self.z)

    __rmul__ = __mul__

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"


class Star:
    def __init__(self, name, radius, mass):
        self.name = name
        self.radius = radius
        self.mass = mass
        self.mu = G * mass


class Planet:
    def __init__(self, idx, name, radius, mass, gravity, h, i, W, w, e, mu):
        self.idx = idx
        self.name = name
        self.radius = radius
        self.mass = mass
        self.gravity = gravity

        self.h = h            # m^2/s,          Specific Angular Momentum
        self.i = i            # radians,        Inclination
        self.W = W            # radians,        RA of Ascending Node
        self.w = w            # radians,        Argument of Periapsis
        self.e = e            # dimensionless,  Eccentricity

        self.mu = mu # gravitational parameter relative to sun

    def position(self, t, max_iter=15): # where t is time
        T = ( (2*PI)/(self.mu**2) ) * (self.h/math.sqrt(1-self.e**2))**3 # period of the orbit
        M = (2*PI/T) * t # mean anomaly
        E = self.eccentric_anomaly(M, self.e, max_iter)
        theta = 2*math.atan(math.sqrt((1+self.e)/(1-self.e)) * math.tan(E/2))
        r = (self.h**2 / self.mu) * (1 / (1 + self.e*math.cos(theta)))
        return self.orbital_to_cartesian(r, theta)

    @staticmethod
    def eccentric_anomaly(M, e, max_iter=15):
        E = M
        # bessel function
        for n in range(1, max_iter):
            J = sum(((-1)**k / (math.factorial(k)*math.factorial(n+k))) * ((n*e)/2)**(n+2*k) for k in range(max_iter-1))
            E += (2/n) * J * math.sin(n*M)
        return E

    def orbital_to_cartesian(self, r, theta):
        w, W, i = self.w, self.W, self.i
        x = r*(math.cos(theta)*(math.cos(W)*math.cos(w)-math.sin(W)*math.sin(w)*math.cos(i)) +
               math.sin(theta)*(-math.cos(W)*math.sin(w)-math.sin(W)*math.cos(i)*math.cos(w)))
        y = r*(math.cos(theta)*(math.sin(W)*math.cos(w)+math.cos(W)*math.cos(i)*math.sin(w)) +
               math.sin(theta)*(-math.sin(W)*math.sin(w)+math.cos(W)*math.cos(i)*math.cos(w)))
        z = r*(math.cos(theta)*math.sin(i)*math.sin(w) + math.sin(theta)*math.sin(i)*math.cos(w))
        return Vec3D(x, y, z)

    def __str__(self):
        pr_str = f"\nPLANET {self.idx}"
        pr_str += f"\nName:             \t{self.name}"
        pr_str += f"\nRadius:           \t{self.radius} m"
        pr_str += f"\nMass:             \t{self.mass} kg"

        pr_str += f"\nAngular Momentum: \t{self.h} m^2/s"
        pr_str += f"\nEccentricity:     \t{self.e}"
        pr_str += f"\nInclination:      \t{self.i} rad ({round(rad2deg(self.i),2)} deg)"
        pr_str += f"\nRA of Node:       \t{self.W} rad ({round(rad2deg(self.W),2)} deg)"
        pr_str += f"\nArg. of Periapsis:\t{self.w} rad ({round(rad2deg(self.w),2)} deg)"
        
        pr_str += f"\nGravitational Para.: \t{self.mu} m^3/s^2"

        return pr_str



# FUNCTIONS ----------------------------------------------------------------------------------------
def load_planets_from_csv(filename, sun):
    planets = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)
        idx = 1
        for row in reader:
            mu = G * (sun.mass + float(row[3]))
            planet = Planet(
                idx, 
                row[0], 
                float(row[1])*1000, 
                float(row[3]), 
                float(row[5]), 
                float(row[19]), 
                float(row[8]), 
                float(row[10]), 
                float(row[12]), 
                float(row[13]), 
                mu)
            planets.append(planet)
            idx += 1
    return planets


def parametric_orbit_x(t, r_p, a, b, w, W, i):
    return ( 
        ((r_p - a) + a*np.cos(t))*(np.cos(W)*np.cos(w)-np.sin(W)*np.sin(w)*np.cos(i)) 
        + (b*np.sin(t))*(-np.cos(W)*np.sin(w)-np.sin(W)*np.cos(i)*np.cos(w))
    )

def parametric_orbit_y(t, r_p, a, b, w, W, i):
    return (
        ((r_p - a) + a*np.cos(t))*(np.sin(W)*np.cos(w)+np.cos(W)*np.cos(i)*np.sin(w))
        + (b*np.sin(t))*(-np.sin(W)*np.sin(w)+np.cos(W)*np.cos(i)*np.cos(w))
    )

def parametric_orbit_z(t, r_p, a, b, w, W, i):
    return (
        ((r_p - a) + a*np.cos(t))*np.sin(i)*np.sin(w) 
        + (b*np.sin(t))*np.sin(i)*np.cos(w)
    )


def true_anomaly_from_state(r, v, mu):
    h = r.cross(v)
    e_vec = (1/mu) * ((v.mag()**2 - mu/r.mag())*r - (r.dot(v))*v)
    e = e_vec.mag()

    cos_nu = e_vec.dot(r)/(e*r.mag())
    cos_nu = max(-1.0, min(1.0, cos_nu))
    nu = math.acos(cos_nu)

    # determine sign using radial velocity
    if r.dot(v) < 0:
        nu = 2*math.pi - nu

    return nu


def hyperbolic_trajectory(): # outputs orbital elements
    print("Running hyperbolic trajectory solver...")

    Dt = TRAVEL_TIME
    mu_sun = G*sun.mass # gravitational parameter of sun

    r_1 = planets[P_DEP-1].position(t_0)
    r_2 = planets[P_ARR-1].position(t_0+TRAVEL_TIME)


    # check if hyperbolic
    c = (r_2 - r_1).mag()
    s = (r_1.mag() + r_2.mag() + c)/2
    Dt_min = math.sqrt(2/mu_sun) * ( s**(3/2) - (s - c)**(3/2) )
    if Dt > Dt_min:
        print("NOTE: TRAJECTORY IS NOT HYPERBOLIC")
    else:
        print("Trajectory is hyperbolic")


    # change in true anomaly
    cos_Dtheta = r_1.dot(r_2)/(r_1.mag()*r_2.mag())
    cos_Dtheta = max(-1.0, min(1.0, cos_Dtheta))
    D_theta = math.acos(cos_Dtheta)
    print(f"D_theta = {D_theta}")
    cross_z = r_1.cross(r_2).z  # prograde check
    if cross_z < 0:
        D_theta = 2*math.pi - D_theta
        print("Trajectory was not prograde")
    else:
        print("Trajectory is prograde")

    # A parameter
    A = math.sin(D_theta) * math.sqrt( (r_1.mag()*r_2.mag()) / (1 - math.cos(D_theta)) )
    print(f"A = {A}")

    # Stumpff Functions
    def C(z):
        if z > 0:   return (1 - math.cos(math.sqrt(z))) / z
        elif z < 0: return (math.cosh(math.sqrt(-z)) - 1) / (-z)
        else:       return (0.5)

    def S(z):
        if z > 0:   return (math.sqrt(z) - math.sin(math.sqrt(z))) / (z**(3/2))
        elif z < 0: return (math.sinh(math.sqrt(-z)) - math.sqrt(-z)) / ((-z)**(3/2))
        else:       return (1/6)

    # y(z) function
    def y(z):
        return r_1.mag() + r_2.mag() + A * ( (z*S(z) - 1)/math.sqrt(C(z)) )

    # iterative solving of universal variables via bisection
    z_low = -4*math.pi**2
    z_high = 4*math.pi**2
    tol = 1e-8 # tolerance
    max_iter = 1000

    for _ in range(max_iter):
        z = 0.5 * (z_low + z_high)
        if y(z) < 0:
            z_low = z
            continue
        t_z = ( (y(z)/C(z))**1.5 * S(z) + A * math.sqrt(y(z)) ) / math.sqrt(mu_sun)
        if abs(t_z - Dt) < tol:
            break
        if t_z > Dt:
            z_high = z
        else:
            z_low = z
    # else:
    #     raise RuntimeError("Lambert hyperbolic solver failed to converge")

    print(f"z converged to {z}")

    # lagrange coefficients
    f = 1 - y(z)/r_1.mag()
    g = A * math.sqrt(y(z)/mu_sun)
    g_dot = 1 - y(z)/r_2.mag()

    # velocity vectors
    v_1 = (r_2 - r_1*f) * (1/g)
    v_2 = (r_2*g_dot - r_1) * (1/g)

    
    # trajectory elements
    h = r_1.cross(v_1)
    i = 2*PI - math.acos(h.z / h.mag())

    N = Vec3D( (-1*h.y), (-1*h.x), 0 )

    W = 0
    if N.y >= 0:
        W = math.acos(N.x/N.mag())
    else:
        W = 2*PI - math.acos(N.x/N.mag())

    v_1_r = (r_1.dot(v_1))/r_1.mag() # radial velocity

    e = (1/mu_sun)*( (v_1.mag()**2 - (mu_sun/r_1.mag()))*r_1 - (r_1.mag()*v_1_r*v_1) )

    w = 2*PI
    if e.z >= 0:
        w -= math.acos( N.dot(e)/(N.mag()*e.mag()) )
    else:
        w -= 2*PI - math.acos( N.dot(e)/(N.mag()*e.mag()) )

    # print velocity information
    a_1 = ((planets[P_DEP-1].h**2)/planets[P_DEP-1].mu)*(1/(1-planets[P_DEP-1].e**2))
    a_2 = ((planets[P_ARR-1].h**2)/planets[P_ARR-1].mu)*(1/(1-planets[P_ARR-1].e**2))

    v_he_departure = v_1.mag() - math.sqrt( 
        planets[P_DEP-1].mu * ( 2/r_1.mag() - 1/a_1) 
        )

    v_he_arrival = math.sqrt( 
        planets[P_ARR-1].mu * ( 2/r_2.mag() - 1/a_2)
        ) - v_2.mag()

    print(f"{planets[P_DEP-1].name} to {planets[P_ARR-1].name} in {TRAVEL_DAYS} days")
    print("-------------------------------------------------------------")

    print("Hyperbolic Excess Speeds:")
    print(f" - Departure:  {round(v_he_departure)} m/s")
    print(f" - Arrival:   {round(v_he_arrival)} m/s")

    return h.mag(), e.mag(), i, W, w, r_1, r_2, v_1, v_2, v_he_departure, v_he_arrival


def hyperbolic_orbit_xyz(nu, h, e, mu, w, W, i):
    r = (h**2 / mu) / (1 + e*np.cos(nu))

    x_pf = r * np.cos(nu)
    y_pf = r * np.sin(nu)

    cosW, sinW = np.cos(W), np.sin(W)
    cosw, sinw = np.cos(w), np.sin(w)
    cosi, sini = np.cos(i), np.sin(i)

    x = (cosW*cosw - sinW*sinw*cosi)*x_pf + (-cosW*sinw - sinW*cosw*cosi)*y_pf
    y = (sinW*cosw + cosW*sinw*cosi)*x_pf + (-sinW*sinw + cosW*cosw*cosi)*y_pf
    z = (sinw*sini)*x_pf + (cosw*sini)*y_pf

    return x, y, z


# MAIN ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    sun = Star("82 G. Eridani", 647001000, 1.5900e30)
    planets = load_planets_from_csv('eri_planets.csv', sun)


    # PARAMETERS ---------------------------------------------------------------
    P_DEP, P_ARR = 2, 8
    TRAVEL_DAYS = 90
    TRAVEL_TIME = TRAVEL_DAYS*(24*3600)
    t_0 = 746000000

    ALTITUDE_DEP = 5e7 # altitude of circular parking orbit for departure
    ALTITUDE_ARR = 5e7 # altitude of periapsis of arrival orbit
    ECC_ARR = 0.4 # eccentricity of arrival orbit

    SHOW_FULL_TRAJECTORY = False
    # --------------------------------------------------------------------------


    # Planet position arrays
    p_pos = []
    for planet in planets:
        if planet.idx != P_DEP or planet.idx != P_ARR:
            p_pos.append(planet.position(t_0))

    p_pos_departure = []
    p_pos_departure.append(planets[P_DEP-1].position(t_0))
    p_pos_departure.append(planets[P_ARR-1].position(t_0))
    
    p_pos_arrival = []
    p_pos_arrival.append(planets[P_DEP-1].position(t_0+TRAVEL_TIME))
    p_pos_arrival.append(planets[P_ARR-1].position(t_0+TRAVEL_TIME))


    # Hyperbolic trajectory elements
    hyp = hyperbolic_trajectory()
    h_Tr = hyp[0]
    e_Tr = hyp[1]
    i_Tr = hyp[2]
    W_Tr = hyp[3]
    w_Tr = hyp[4]
    mu_sun = G*sun.mass

    print("Trajectory Elements:")
    print(f" - Angular Mom.:    {round(h_Tr)} m^2/s")
    print(f" - Eccentricity:    {e_Tr}")
    print(f" - Inclination:     {round(i_Tr * (180/PI))} deg")
    print(f" - RA of Asc. Node: {round(W_Tr * (180/PI))} deg")
    print(f" - Arg. of Peri.:   {round(w_Tr * (180/PI))} deg")

    a_Tr = ((h_Tr**2)/mu_sun)*(1/(e_Tr**2 - 1))
    b_Tr = a_Tr*math.sqrt(e_Tr**2 - 1)
    r_p_Tr = ((h_Tr**2)/mu_sun)*(1/(1 + e_Tr))

    # DEPARTURE DELTA-V
    v_he_dep = hyp[9]

    mu_dep = G * planets[P_DEP-1].mass # grav p of spacecraft around departure planet.
    r_p_dep = planets[P_DEP-1].radius + ALTITUDE_DEP # periapsis radius of departure, also height of circular parking orbit.

    h_D = r_p_dep * math.sqrt( v_he_dep**2 + (2*mu_dep)/(r_p_dep) )
    v_p_dep = h_D/r_p_dep
    v_c_dep = math.sqrt( mu_dep / r_p_dep )

    delta_v_departure = v_p_dep - v_c_dep


    # ARRIVAL DELTA-V
    v_he_arr = hyp[10]

    mu_arr = G * planets[P_ARR-1].mass
    r_p_arr = planets[P_ARR-1].radius + ALTITUDE_ARR

    v_p_hyperbola_arr = math.sqrt( v_he_arr**2 + (2*mu_arr)/(r_p_arr) )
    v_p_capture_arr = math.sqrt( (mu_arr*(1+ECC_ARR))/r_p_arr )

    delta_v_arrival = v_p_hyperbola_arr - v_p_capture_arr

    print("Delta-v:")
    print(f" - Departure:  {round(delta_v_departure)} m/s")
    print(f" - Arrival:    {round(delta_v_arrival)} m/s")
    print(f" - Total:      {round(delta_v_departure + delta_v_arrival)} m/s")


    # Plots --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    xmin, xmax, ymin, ymax = -120000, 120000, -120000, 120000
    ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')
    ax.xaxis.set_major_formatter(FuncFormatter(scale_formatter_sci_km))
    ax.yaxis.set_major_formatter(FuncFormatter(scale_formatter_sci_km))
    # ax.set_xlabel("x (km)")
    # ax.set_ylabel("y (km)")


    # Non-trajectory planets
    for pos in p_pos:
        ax.scatter(pos.x/S_FAC, pos.y/S_FAC, color='0.2')
    # Planet positions at departure (blue)
    for pos in p_pos_departure:
        ax.scatter(pos.x/S_FAC, pos.y/S_FAC, color='blue')
    # Planet positions at arrival (red)
    for pos in p_pos_arrival:
        ax.scatter(pos.x/S_FAC, pos.y/S_FAC, color='red')
    # Sun
    ax.scatter(0,0,color='orange')


    # Planet Orbits
    t_range = np.arange(0, 2*PI, PI/256)
    for p in planets:
        a = ((p.h**2)/p.mu)*(1/(1-p.e**2)) # semimajor axis
        b = a*math.sqrt(1-p.e**2) # semiminor axis
        r_p = ((p.h**2)/p.mu)*(1/(1+p.e)) # periapsis radius

        plt.plot(
            parametric_orbit_x(t_range, r_p/S_FAC, a/S_FAC, b/S_FAC, p.w, p.W, p.i),
            parametric_orbit_y(t_range, r_p/S_FAC, a/S_FAC, b/S_FAC, p.w, p.W, p.i),
            "0.8", zorder=0
            )

    # Plot hyperbolic trajectory
    if SHOW_FULL_TRAJECTORY:
        nu_max = math.acos(-1/e_Tr)
        nu = np.linspace(-nu_max, nu_max, 10000)

        x, y, z = hyperbolic_orbit_xyz(
            nu,
            h_Tr,
            e_Tr,
            mu_sun,
            w_Tr,
            W_Tr,
            i_Tr
        )

        plt.plot(x/S_FAC, y/S_FAC, 'r', linewidth=2, zorder=3)
    else:
        r_1 = hyp[5]
        r_2 = hyp[6]
        v_1 = hyp[7]
        v_2 = hyp[8]
        nu_dep = true_anomaly_from_state(r_1, v_1, mu_sun)
        nu_arr = true_anomaly_from_state(r_2, v_2, mu_sun)
        if nu_arr < nu_dep:
            nu_arr += 2*math.pi
        nu_plot = np.linspace(nu_dep, nu_arr, 5000)

        x, y, z = hyperbolic_orbit_xyz(
            nu_plot,
            h_Tr,
            e_Tr,
            mu_sun,
            w_Tr,
            W_Tr,
            i_Tr
        )

        plt.plot(x/S_FAC, y/S_FAC, 'r', linewidth=2, zorder=3)
    

    ax.set_title(f"Hyperbolic Transfer from {planets[P_DEP-1].name} to {planets[P_ARR-1].name} in {TRAVEL_DAYS} days")
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()
