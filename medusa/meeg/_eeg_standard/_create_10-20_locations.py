from medusa import meeg
from medusa.plots.head_plots import TopographicPlot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import sympy as sp


class EEGMontageHelper:
    def __init__(self):
        self.channels = dict()

    def add_channel(self, label, x, y):
        if label != "":
            self.channels[label] = (x, y)

    def save(self, file_path):
        with open(file_path, 'w') as file:
            file.write("label\tx\ty")
            for label, coords in self.channels.items():
                file.write("\n%s\t%.4f\t%.4f" % (label.upper(), coords[0],
                                                 coords[1]))
        print(f"Data saved to {file_path}")

def get_circle_three_points(p1_coords, p2_coords, p3_coords):
    """
    Computes a circle whose circumference crosses three points.
    :param p1_coords: (x,y) coordinates of the first point
    :param p2_coords: (x,y) coordinates of the second point
    :param p3_coords: (x,y) coordinates of the third point
    :return: x_circle, y_circle, [r, h, k]
        r: radius of the circle
        cx: x coordinate of the center of the circle
        cy: y coordinate of the center of the circle
    """
    # Define the symbols for the coefficients D, E, F and point coordinates
    D, E, F = sp.symbols('D E F')
    p1x, p1y, p2x, p2y, p3x, p3y = sp.symbols('p1x p1y p2x p2y p3x p3y')

    # Equations based on the general circle equation x^2 + y^2 + Dx + Ey + F = 0
    eq1 = sp.Eq(p1x ** 2 + p1y ** 2 + D * p1x + E * p1y + F, 0)
    eq2 = sp.Eq(p2x ** 2 + p2y ** 2 + D * p2x + E * p2y + F, 0)
    eq3 = sp.Eq(p3x ** 2 + p3y ** 2 + D * p3x + E * p3y + F, 0)

    # Solve the system of equations for D, E, and F
    solution = sp.solve([eq1, eq2, eq3], (D, E, F))

    # Substitute the values of p1, p2, and p3 into the expressions for D, E, and F
    D_val = solution[D].subs({p1x: p1_coords[0], p1y: p1_coords[1],
                              p2x: p2_coords[0], p2y: p2_coords[1],
                              p3x: p3_coords[0], p3y: p3_coords[1]})
    E_val = solution[E].subs({p1x: p1_coords[0], p1y: p1_coords[1],
                              p2x: p2_coords[0], p2y: p2_coords[1],
                              p3x: p3_coords[0], p3y: p3_coords[1]})
    F_val = solution[F].subs({p1x: p1_coords[0], p1y: p1_coords[1],
                              p2x: p2_coords[0], p2y: p2_coords[1],
                              p3x: p3_coords[0], p3y: p3_coords[1]})

    # Calculate the center (h, k) and radius r of the circle
    h = -D_val / 2
    k = -E_val / 2
    r = float(sp.sqrt(h ** 2 + k ** 2 - F_val))

    # Convert the symbolic values of h and k to numerical values
    cx = float(h)
    cy = float(k)
    return r, cx, cy

def divide_arc(pvert, p2, r, c1, c2, ratio=0.5, right=True, anterior=True):
    """
    :param pvert: sagital point (x,y)
    :param p2: end arc point (x,y)
    :param r: circle radius
    :param c1: x center of the circle
    :param c2: y center of the circle
    :param ratio: ratio of division
    :param right: true if right hemisphere, false otherwise
    :param anterior: true if anterior, false is posterior
    :return: (x,y) coordinates of the point
    """
    alpha = np.arcsin((p2[0] - pvert[0])/r)
    offset = 1.5 if anterior else 0.5
    if right:
        x1 = c1 + r * np.cos(offset * np.pi + alpha * ratio)
        y1 = c2 + r * np.sin(offset * np.pi + alpha * ratio)
    else:
        x1 = c1 + r * np.cos(offset * np.pi - alpha * ratio)
        y1 = c2 + r * np.sin(offset * np.pi - alpha * ratio)
    return x1, y1

def plot_circle(ax, r, cx, cy):
    theta = np.linspace(0, 2 * np.pi, 100)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    ax.plot(x, y, 'k')

# Function to handle plotting, channel addition, and annotation for a given ratio
def add_channel_label(point_start, point_end, ratio, label, right, anterior):
    x, y = divide_arc(point_start, point_end, r, c1, c2, ratio=ratio,
                      right=right, anterior=anterior)
    ax.scatter(x, y, 20, 'k')
    eeg10_20.add_channel(label, x, y)
    ax.text(x, y, label)


# Initialization
fig, ax = plt.subplots(1)
eeg10_20 = EEGMontageHelper()

# Outer circle
plot_circle(ax, r=1, cx=0, cy=0)

# Inner circle
r = 4/5
plot_circle(ax, r=r, cx=0, cy=0)
px = r*np.cos(np.arange(0, 2*np.pi, np.pi/5))
py = r*np.sin(np.arange(0, 2*np.pi, np.pi/5))
ax.scatter(px, py, 20, 'k')
labels = ["T8", "F8", "Fp2", "Fp1", "F7",
          "T7", "P7", "O1", "O2", "P8"]
for i in zip(labels, px, py):
    eeg10_20.add_channel(i[0], float(i[1]), float(i[2]))
    ax.text(i[1], i[2], i[0])

# Sagital line
ax.plot([0,0], [-1, 1], 'k')
py = np.array([1, 0.4, 0, -0.4, -1])
px = np.zeros((py.shape))
ax.scatter(px, py, 20, 'k')
labels = ["Nz", "Fz", "Cz", "Pz", "Iz"]
for i in zip(labels, px, py):
    eeg10_20.add_channel(i[0], float(i[1]), float(i[2]))
    ax.text(i[1], i[2], i[0])

# Coronal line
ax.plot([-1, 1], [0,0], 'k')
px = np.arange(-0.8, 0.8+0.4, 0.4)
py = np.zeros((px.shape))
ax.scatter(px, py, 20, 'k')
labels = ["T7", "C3", "Cz", "C4", "T8"]
for i in zip(labels, px, py):
    eeg10_20.add_channel(i[0], float(i[1]), float(i[2]))
    ax.text(i[1], i[2], i[0])

# F7 arc
p1 = (4/5 * np.cos(np.pi - 2 * np.pi/10), 4/5 * np.sin(np.pi - 2 * np.pi/10))
p2 = (0, 2/5)
p3 = (4/5 * np.cos(2 * np.pi/10), 4/5 * np.sin(2 * np.pi/10))
r, c1, c2 = get_circle_three_points(p1, p2, p3)
plot_circle(ax, r=r, cx=c1, cy=c2)
right_channels = [('F4', 0.5)]
for label, ratio in right_channels:
    add_channel_label(p2, p3, ratio, label, right=False, anterior=True)
left_channels = [('F3', 0.5)]
for label, ratio in left_channels:
    add_channel_label(p2, p3, ratio, label, right=True, anterior=True)

# P7 arc
p1 = (4/5 * np.cos(np.pi + 2 * np.pi/10), 4/5 * np.sin(np.pi + 2 * np.pi/10))
p2 = (0, -2/5)
p3 = (4/5 * np.cos(-2 * np.pi/10), 4/5 * np.sin(-2 * np.pi/10))
r, c1, c2 = get_circle_three_points(p1, p2, p3)
plot_circle(ax, r=r, cx=c1, cy=c2)
right_channels = [('P4', 0.5)]
for label, ratio in right_channels:
    add_channel_label(p2, p3, ratio, label, right=False, anterior=False)
left_channels = [('P3', 0.5)]
for label, ratio in left_channels:
    add_channel_label(p2, p3, ratio, label, right=True, anterior=False)

# Ears and mastoid
a1 = (-1.1, 0.1)
m1 = (-1.1, -0.1)
a2 = (1.1, 0.1)
m2 = (1.1, -0.1)
ears = [("A1", a1), ("A2", a2), ("M1", m1), ("M2", m2)]
ax.scatter([a1[0], a2[0], m1[0], m2[0]],
           [a1[1], a2[1], m1[1], m2[1]], 20, 'k')
for i in ears:
    eeg10_20.add_channel(i[0], float(i[1][0]), float(i[1][1]))
    ax.text(i[1][0], i[1][1], i[0])

# Plot
ax.set_xticks(np.arange(-1, 1.2, 0.2))
ax.set_yticks(np.arange(-1, 1.2, 0.2))
ax.set_aspect(1)
ax.grid(True)
ax.set_xlim((-1.2, 1.2))
ax.set_ylim((-1.2, 1.2))
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.2))
plt.show()

# Save
print(eeg10_20.channels)
eeg10_20.save("10-20.tsv")
