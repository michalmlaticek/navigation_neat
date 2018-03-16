from PIL import Image
import random
import math
import numpy as np


def generate_voronoi_diagram(width, height, num_cells, map_id):
    map_image = Image.new("RGB", (width, height))
    map_matrix = np.zeros((width, height))
    putpixel = map_image.putpixel
    imgx, imgy = map_image.size
    nx = []
    ny = []
    nr = []
    ng = []
    nb = []
    for i in range(num_cells):
        nx.append(random.randrange(imgx))
        ny.append(random.randrange(imgy))
        nr.append(random.randrange(10) - 1)  # 1 to 10 probability that it's going to be black (isle)
        # ng.append(random.randrange(1))
        # nb.append(random.randrange(1))
    for y in range(imgy):
        for x in range(imgx):
            dmin = math.hypot(imgx - 1, imgy - 1)
            j = -1
            for i in range(num_cells):
                d = math.hypot(nx[i] - x, ny[i] - y)
                if d < dmin:
                    dmin = d
                    j = i
            color = (0, 0, 0)
            c = 0
            if nr[j] > 0:
                c = 1
                color = (255, 255, 255)
            map_matrix[x, y] = c
            putpixel((x, y), color)

    np.savetxt("maps/map_{}.csv".format(map_id), map_matrix.astype(int), fmt="%i", delimiter=",")
    np.save("maps/map_{}.npy".format(map_id), map_matrix.astype(int), allow_pickle=False)
    map_image.save("maps/map_{}.png".format(map_id), "PNG")
    # map_image.show()


for i in range(0, 100):
    generate_voronoi_diagram(250, 250, 15, i)
