import csv
import math
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# CONSTANTS
# ----------------------
G = 6.67430e-11      # m^3/kg/s^2
PI = math.pi
S_FAC = 2e8          # scale for plotting

# ----------------------
# VECTOR CLASS
# ----------------------
class Vec3D:
    def __init__(self, x, y, z):
        self.v = np.array([x, y, z], dtype=float)

    def magnitude(self):
        return np.linalg.norm(self.v)

    def dot(self, other):
        return np.dot(self.v, other.v)

    def cross(self, other):
        return Vec3D(*np.cross(self.v, other.v))

    def __add__(self, other):
        return Vec3D(*(self.v + other.v))

    def __sub__(self, other):
        return Vec3D(*(self.v - other.v))

    def __mul__(self, scalar):
        return Vec3D(*(self.v * scalar))

    __rmul__ = __mul__

    def __str__(self):
        return f"({self.v[0]}, {self.v[1]}, {self.v[2]})"

# ----------------------
# STAR & PLANET CLASSES
# ----------------------
class Star:
    def __init__(self, name, radius, mass):
        self.name = name
        self.radius = radius
        self.mass = mass
        self.mu = G * mass

class Planet:
    def __init__(self, idx, name, radius, mass, ang_mom, inc, ra_an, arg_p, ecc, mu_sun):
        self.idx = idx
        self.name = name
        self.radius = radius
        self.mass = mass
        self.ang_mom = ang_mom
        self.inclination = inc
        self.ra_an = ra_an
        self.arg_p = arg_p
        self.ecc = ecc
        self.mu_sun = mu_sun

    def position(self, time, max_iter=15):
        T = (2*PI)/(self.mu_sun**2) * (self.ang_mom/np.sqrt(1-self.ecc**2))**3
        M = (2*PI/T) * time
        E = self.eccentric_anomaly(M, self.ecc, max_iter)
        theta = 2*math.atan(math.sqrt((1+self.ecc)/(1-self.ecc)) * math.tan(E/2))
        r = (self.ang_mom**2 / self.mu_sun) * (1 / (1 + self.ecc*math.cos(theta)))
        return self.orbital_to_cartesian(r, theta)

    @staticmethod
    def eccentric_anomaly(M, e, max_iter=15):
        E = M
        for n in range(1, max_iter):
            J = sum(((-1)**k / (math.factorial(k)*math.factorial(n+k))) * ((n*e)/2)**(n+2*k) for k in range(max_iter-1))
            E += (2/n) * J * math.sin(n*M)
        return E

    def orbital_to_cartesian(self, r, theta):
        w, W, i = self.arg_p, self.ra_an, self.inclination
        x = r*(math.cos(theta)*(math.cos(W)*math.cos(w)-math.sin(W)*math.sin(w)*math.cos(i)) +
               math.sin(theta)*(-math.cos(W)*math.sin(w)-math.sin(W)*math.cos(i)*math.cos(w)))
        y = r*(math.cos(theta)*(math.sin(W)*math.cos(w)+math.cos(W)*math.cos(i)*math.sin(w)) +
               math.sin(theta)*(-math.sin(W)*math.sin(w)+math.cos(W)*math.cos(i)*math.cos(w)))
        z = r*(math.cos(theta)*math.sin(i)*math.sin(w) + math.sin(theta)*math.sin(i)*math.cos(w))
        return Vec3D(x, y, z)

# ----------------------
# CSV LOADING
# ----------------------
def load_planets_from_csv(filename, sun):
    planets = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)
        idx = 1
        for row in reader:
            mu_sun = G * (sun.mass + float(row[3]))
            planet = Planet(
                idx,
                row[0],                   # name
                float(row[1])*1000,       # radius (km->m)
                float(row[3]),             # mass
                float(row[5]),             # angular momentum
                float(row[8]),             # inclination
                float(row[10]),            # RA node
                float(row[12]),            # argument of periapsis
                float(row[13]),            # eccentricity
                mu_sun
            )
            planets.append(planet)
            idx += 1
    return planets

# ----------------------
# TRUE HYPERBOLIC TRAJECTORY
# ----------------------
def hyperbolic_conic(r1, r2, mu, steps=500):
    """Compute hyperbolic trajectory given r1, r2, and Sun mu."""
    r1_vec = r1.v
    r2_vec = r2.v
    # Approximate v_inf using straight line magnitude (simplification)
    v_inf = (r2_vec - r1_vec)/1e7  # adjust scale for plotting

    h_vec = np.cross(r1_vec, v_inf)
    h_mag = np.linalg.norm(h_vec)
    r1_mag = np.linalg.norm(r1_vec)
    v1_mag = np.linalg.norm(v_inf)
    e_vec = (np.cross(v_inf, h_vec)/mu) - r1_vec/r1_mag
    e_mag = np.linalg.norm(e_vec)

    # Semi-major axis
    a = -mu/(v1_mag**2)
    theta = np.linspace(-np.arccos(1/e_mag), np.arccos(1/e_mag), steps)
    r = a*(e_mag**2 - 1)/(1 + e_mag*np.cos(theta))
    # rotate to orbital plane
    # For visualization, approximate plane as xy
    x = r*np.cos(theta) + r1_vec[0]
    y = r*np.sin(theta) + r1_vec[1]
    return x/S_FAC, y/S_FAC

# ----------------------
# MAIN PROGRAM
# ----------------------
if __name__ == "__main__":
    sun = Star("82 G. Eridani", 647001000, 1.5900e30)
    planets = load_planets_from_csv('eri_planets.csv', sun)

    P_DEP, P_ARR = 1, 2
    TRAVEL_DAYS = 90
    TRAVEL_TIME = TRAVEL_DAYS*24*3600
    t0 = 0

    # Positions at departure and arrival
    pos_dep_time = [p.position(t0) for p in planets]
    pos_arr_time = [p.position(t0 + TRAVEL_TIME) for p in planets]

    # Hyperbolic trajectory
    r1 = planets[P_DEP-1].position(t0)
    r2 = planets[P_ARR-1].position(t0 + TRAVEL_TIME)
    traj_x, traj_y = hyperbolic_conic(r1, r2, sun.mu)

    # ----------------------
    # PLOTTING
    # ----------------------
    fig, ax = plt.subplots(figsize=(10,10))

    # Planet orbits
    theta = np.linspace(0, 2*PI, 500)
    for p in planets:
        a = (p.ang_mom**2 / p.mu_sun)*(1/(1-p.ecc**2))
        b = a*math.sqrt(1-p.ecc**2)
        r_p = (p.ang_mom**2 / p.mu_sun)*(1/(1+p.ecc))
        orbit_x = r_p + a*np.cos(theta)
        orbit_y = r_p + b*np.sin(theta)
        ax.plot(orbit_x/S_FAC, orbit_y/S_FAC, '0.8', zorder=0)

    # Planets at departure (blue)
    for pos, p in zip(pos_dep_time, planets):
        ax.scatter(pos.v[0]/S_FAC, pos.v[1]/S_FAC, color='blue', s=60, label=f"{p.name} t0")
    # Planets at arrival (red)
    for pos, p in zip(pos_arr_time, planets):
        ax.scatter(pos.v[0]/S_FAC, pos.v[1]/S_FAC, color='red', s=60, label=f"{p.name} t1")

    # Hyperbolic transfer trajectory
    ax.plot(traj_x, traj_y, 'r--', linewidth=2, label="Transfer trajectory")

    ax.scatter(0,0,color='orange',s=200,label=sun.name)
    ax.set_aspect('equal')
    ax.set_xlabel("X (scaled)")
    ax.set_ylabel("Y (scaled)")
    ax.set_title(f"True Hyperbolic Transfer {planets[P_DEP-1].name} -> {planets[P_ARR-1].name}")
    ax.legend(fontsize=8)
    plt.show()
