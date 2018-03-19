from PIL import Image
import random
import math
import numpy as np


def generate_voronoi_diagram(size, num_cells):
    map_image = Image.new("RGB", (size, size))
    putpixel = map_image.putpixel
    imgx, imgy = map_image.size
    nx = []
    ny = []
    nrgb = []
    for i in range(num_cells):
        nx.append(random.randrange(imgx))
        ny.append(random.randrange(imgy))
        nrgb.append(random.randrange(10) - 1)  # 1 to 10 probability that it's going to be black (isle)
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
            if nrgb[j] > 0:
                color = (255, 255, 255)
            putpixel((x, y), color)

    return map_image


def generate_voronoi_map(size, num_cells, robot_radius, map_folder, map_id):
    map_image = generate_voronoi_diagram(size, num_cells)

    #  add fence
    fence_width = 2
    for k in range(0, size):
        for n in range(0, fence_width):
            map_image.putpixel((k, n), (0, 0, 0))
            map_image.putpixel((n, k), (0, 0, 0))
            map_image.putpixel((k, size - (n + 1)), (0, 0, 0))
            map_image.putpixel((size - (n + 1), k), (0, 0, 0))

    # # clear start and end area
    # areas_to_clear = [(50, 50), (200, 200)]
    # for (x, y) in areas_to_clear:
    #     for m in range(x - robot_radius, x + robot_radius):
    #         for n in range(y - robot_radius, y + robot_radius):
    #             map_matrix[m, n] = 1
    #             map_image.putpixel((m, n), (255, 255, 255))

    # create versions with different paths
    path1_img = Image.open("paths/path1.png").convert("RGB")
    path2_img = Image.open("paths/path2.png").convert("RGB")
    path3_img = Image.open("paths/path3.png").convert("RGB")

    map_path1_img = map_image.copy()
    map_path2_img = map_image.copy()
    map_path3_img = map_image.copy()

    for i in range(0, size):
        for j in range(0, size):
            if path1_img.getpixel((i, j)) == (0, 0, 0):
                map_path1_img.putpixel((i, j), (255, 255, 255))

            if path2_img.getpixel((i, j)) == (0, 0, 0):
                map_path2_img.putpixel((i, j), (255, 255, 255))

            if path3_img.getpixel((i, j)) == (0, 0, 0):
                map_path3_img.putpixel((i, j), (255, 255, 255))

    # save maps
    map_path1_img.save("{}/map_{}_path_1.png".format(map_folder, map_id), "PNG")
    map_path2_img.save("{}/map_{}_path_2.png".format(map_folder, map_id), "PNG")
    map_path3_img.save("{}/map_{}_path_3.png".format(map_folder, map_id), "PNG")

    map_path1_flip = map_path1_img.transpose(Image.FLIP_LEFT_RIGHT)
    map_path2_flip = map_path2_img.transpose(Image.FLIP_LEFT_RIGHT)
    map_path3_flip = map_path3_img.transpose(Image.FLIP_LEFT_RIGHT)

    map_path1_flip.save("{}/map_{}_path_1_flip.png".format(map_folder, map_id), "PNG")
    map_path2_flip.save("{}/map_{}_path_2_flip.png".format(map_folder, map_id), "PNG")
    map_path3_flip.save("{}/map_{}_path_3_flip.png".format(map_folder, map_id), "PNG")

    # np.savetxt("{}/map_{}.csv".format(map_folder, map_id), map_matrix.astype(int), fmt="%i", delimiter=",")
    # np.save("{}/map_{}.npy".format(map_folder, map_id), map_matrix.astype(int), allow_pickle=False)


for i in range(0, 100):
    generate_voronoi_map(250, 15, 20, "maps5", i)
