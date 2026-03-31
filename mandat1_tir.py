import matplotlib.pyplot as plt
import numpy as np
import math

def carlo_monte(x_mean = 0, y_mean = 0, x_std = 1, y_std = 1):
    n_points = 1000000
    points_dans_le_cercle = 0
    points_hors_cercle = 0
    coord_x = np.random.normal(x_mean, x_std, n_points)
    coord_y = np.random.normal(y_mean, y_std, n_points)


    for x, y in zip(coord_x, coord_y):
        if np.sqrt(x**2 + y**2) <= 0.1:
            points_dans_le_cercle += 1
        else:
            points_hors_cercle += 1

    # calcul probabilité d'être dans le cercle
    p_dans_le_cercle = points_dans_le_cercle / n_points
    print(f"Probabilité d'être dans le cercle: {p_dans_le_cercle}")

if __name__ == "__main__":
    carlo_monte(0,0, 0.1, 0.4) # z > 10
    carlo_monte(0,0, 0.1, 0.05) # z < 1
    
    
    