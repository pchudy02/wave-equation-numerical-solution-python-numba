import copy

import pygame
import numpy as np
import random
from scipy.signal import savgol_coeffs
from numba import cuda, float64, int32
import matplotlib as mpl

h = 1  # spatial step width
t = 1  # time step width
dimx = 1001  # width of the simulation domain
dimy = 1001  # height of the simulation domain
cellsize = 0.5  # display size of a cell in pixel
offset = 8
deriv_coffs_space = (np.array(savgol_coeffs(offset * 2 + 1, offset * 2, deriv=2, use='dot'), dtype=np.float64))[offset:]
deriv_coffs_time = np.array(savgol_coeffs(3, 2, deriv=2, use='dot'), dtype=np.float64)
cmap = np.array([np.array(mpl.cm.inferno(i)[:3]) * 256 for i in np.linspace(0, 1, 256)], dtype=np.uint8)
pass
thread_block = (8, 4)


def init_simulation():
    u = np.zeros((3, dimx + offset * 2 + 2, dimy + offset * 2 + 2), dtype=np.float64)
    u[0, u.shape[1] // 2, u.shape[2] // 2] = 1000
    # u[0, dimx // 2, :] = 1000
    # u[0, 9, 9] = 10
    # The three dimensional simulation grid u[time, dimx, dimy]

    c = 0.5  # The "original" wave propagation speed

    alpha = np.zeros((u.shape[1], u.shape[2]), dtype=np.float64)
    # wave propagation velocities of the entire simulation domain

    alpha[:, :] = ((c * t) / h) ** 2  # will be set to a constant value of tau
    # alpha[2 * dimx // 3:2 * dimx // 3 + 10, 0:30] = 0
    # alpha[2 * dimx // 3:2 * dimx // 3 + 10, 30:40] = 0
    # alpha[2 * dimx // 3:2 * dimx // 3 + 10, 60:70] = 0

    return u, alpha


@cuda.jit(f'void(float64[:,:,:], float64[:,:,:], float64[:,:], float64[:])')
def update(u, result, alpha, deriv_coffs):
    x, y = cuda.grid(2)
    idx = cuda.threadIdx.x
    idy = cuda.threadIdx.y

    deriv_coffs_loc = cuda.shared.array(shape=5, dtype=np.float64)

    # shape=(offset * 2 + 1, offset * 2 + 32, offset * 2 + 32)
    u_loc = cuda.shared.array(shape=(5, 12, 8), dtype=np.float64)
    alpha_loc = cuda.shared.array(shape=(8, 4), dtype=np.float64)

    if x < u.shape[1] and y < u.shape[2]:

        i = 0
        while i + idx * 4 + idy < deriv_coffs.shape[0]:
            deriv_coffs_loc[i + idx * 4 + idy] = deriv_coffs[i + idx * 4 + idy]
            i += 32  # cuda.blockDim[0] * cuda.blockDim[1]

        if 0 < x < u.shape[1] - 1 and 0 < y < u.shape[2] - 1:
            alpha_loc[idx, idy] = alpha[x, y]
            # result[0, x, y] = alpha_loc[idx, idy]
            # alpha_loc[idx, idy] = x + y

        i, j = -offset - 4, -offset - 2
        while i + idx < 4 + offset:
            while j + idy < 2 + offset:
                if 0 <= i + x < u.shape[1] and 0 <= j + y < u.shape[2]:
                    k = 1
                    while k < u.shape[0]:
                        u_loc[k, i + idx, j + idy] = u[k - 1, i + x, j + y]
                        k += 1
                    pass

                j += 4
            i += 8

        cuda.syncthreads()
        # u_loc[0, idx, idy] = alpha_loc[idx, idx]
        result[0, x, y] = u_loc[0, idx + offset, idy + offset]

        """With multiplying after second loop throws: {CudaAPIError}[700] Call to cuMemcpyDtoH results in 
        UNKNOWN_CUDA_ERROR"""

        if 0 < x < u.shape[1] - 1 and 0 < y < u.shape[2] - 1:
            u_loc[0, idx + offset, idy + offset] = alpha_loc[idx, idx]
            # result[0, x, y] = u_loc[0, idx + offset, idy + offset]
            # u[0, x, y] = alpha_loc[idx, idx]
            # cuda.syncwarp()

            temp = 0
            i = 0
            while i < offset * 2:
                if 0 < i + x - offset < u.shape[1] - 1 and 0 <= i + idx - offset < 8:
                    temp += (u_loc[int(u.shape[0] // 2), i + idx - offset, idy] * deriv_coffs_loc[i])
                    pass
                i += 1

            # result[0, x, y] = alpha_loc[idx, idx]

            # i = 0
            # while i < offset * 2:
            #     if 0 < i + y - offset < u.shape[1] - 1:
            #         temp += u_loc[int(u.shape[0] // 2), x, i + y - offset] * deriv_coffs_loc[i]
            #     i += 1
            #
            # result[0, x, y] = alpha_loc[idx, idx]
            # This multiplying
            # u_loc[0, idx, idy] = u_loc[0, idx, idy] * temp
            #
            # i = 1
            # while i < offset * 2:
            #     u_loc[0, idx, idy] += u_loc[i, x, y] * (-deriv_coffs_loc[i])
            #     i += 1
            #
            # result[0, x, y] = u_loc[0, idx, idy] / deriv_coffs_loc[0] * 0.995

        # elif (x == 0 or x == u.shape[1] - 1) ^ (y == 0 or y == u.shape[2] - 1):
        #     pass


@cuda.jit(f'void(float64[:,:,:], float64[:,:], float64[:], float64[:])')
def generate_new_frame(u, alpha, deriv_coffs_space, deriv_coffs_time):
    x, y = cuda.grid(2)
    idx = cuda.threadIdx.x
    idy = cuda.threadIdx.y

    u_shared = cuda.shared.array(thread_block, dtype=np.float64)

    if x < u.shape[1] and y < u.shape[2]:

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
        # $$$ Generacja następnej klatki $$$ #
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #

        if 0 < x < u.shape[1] - 1 and 0 < y < u.shape[2] - 1:
            x_plus = True
            x_minus = True
            y_plus = True
            y_minus = True

            temp = 0
            i = 0

            while i < deriv_coffs_space.shape[0]:
                if u.shape[1] - 1 > x + i > 0 and x_plus:
                    if alpha[x + i, y] == 0:
                        x_plus = False
                    else:
                        temp += u[1, x + i, y] * deriv_coffs_space[i]

                if u.shape[1] - 1 > x - i > 0 != i and x_minus:
                    if alpha[x - i, y] == 0:
                        x_minus = False
                    else:
                        temp += u[1, x - i, y] * deriv_coffs_space[i]

                if u.shape[1] - 1 > y + i > 0 and y_plus:
                    if alpha[x, y + i] == 0:
                        y_plus = False
                    else:
                        temp += u[1, x, y + i] * deriv_coffs_space[i]

                if u.shape[1] - 1 > y - i > 0 != i and y_minus:
                    if alpha[x, y - i] == 0:
                        y_minus = False
                    else:
                        temp += u[1, x, y - i] * deriv_coffs_space[i]

                i += 1
                # x_plus = True
                # x_minus = True
                # y_plus = True
                # y_minus = True

            temp *= (alpha[x, y] ** 2)

            i = 1
            while i < deriv_coffs_time.shape[0]:
                temp -= u[i, x, y] * deriv_coffs_time[i]
                i += 1

            if (10 ** (-310)) >= temp / deriv_coffs_time[0] >= -(10 ** (-310)):
                u[0, x, y] = 0
                u_shared[idx, idy] = 0
            else:
                u[0, x, y] = (temp / deriv_coffs_time[0])
                u_shared[idx, idy] = (temp / deriv_coffs_time[0])

        # elif (x == 0 or x == u.shape[1] - 1) ^ (y == 0 or y == u.shape[2] - 1):
        #     pass


@cuda.jit('void(float64[:,:,:], uint8[:,:,:], uint8[:,:], float64, float64[:,:])')
def print_frame(u, pixeldata, cmap, max_val_data, debug):
    x, y = cuda.grid(2)

    if x < pixeldata.shape[0] and y < pixeldata.shape[1]:
        x_low = x * u.shape[1] / pixeldata.shape[0]
        x_high = (x + 1) * u.shape[1] / pixeldata.shape[0]
        y_low = y * u.shape[2] / pixeldata.shape[1]
        y_high = (y + 1) * u.shape[2] / pixeldata.shape[1]

        pixel_value = 0

        ix = x_low
        while ix < x_high:
            if x_low % 1 != 0:
                jx = 1 - (x_low % 1)
            elif ix < x_high and ix + 1 < x_high:
                jx = 1
            else:
                jx = x_high - ix

            iy = y_low
            while iy < y_high:
                if y_low % 1 != 0:
                    jy = 1 - (y_low % 1)
                elif iy < y_high and iy + 1 < y_high:
                    jy = 1
                else:
                    jy = y_high - iy

                    pixel_value += jx * jy * u[0, int(ix // 1), int(iy // 1)]

                iy += jy
            ix += jx

        pixel_value /= (u.shape[1] / pixeldata.shape[0]) * (u.shape[2] / pixeldata.shape[1])
        debug[x, y] = pixel_value

        i = 0
        while i < 3:
            if max_val_data != 0:
                pixeldata[x, y, i] = cmap[int(pixel_value / max_val_data * 127) + 127, i]

            else:
                pixeldata[x, y, i] = cmap[127, i]
            i += 1

    pass


@cuda.jit(f'void(float64[:,:,:], float64, uint8[:,:,:], uint8[:,:])')
# @cuda.jit()
def use_color_map(u, scale, pixeldata, cmap):
    x, y = cuda.grid(2)

    cuda.syncthreads()
    if x < pixeldata.shape[0] and y < pixeldata.shape[1]:

        i = 0
        while i < 3:
            if scale != 0:
                pixeldata[x, y, i] = cmap[int(u[0, x + 1, y + 1] / scale * 127) + 127, i]

            else:
                pixeldata[x, y, i] = cmap[127, i]
            i += 1


def place_raindrops(u):
    if (random.random() < 0.1) and True:
        x = random.randrange(5, dimx - 5)
        y = random.randrange(5, dimy - 5)
        u[0, x - 2:x + 2, y - 2:y + 2] = 255


def main():
    pygame.init()
    display = pygame.display.set_mode((dimx * cellsize, dimy * cellsize))
    pygame.display.set_caption("Solving the 2d Wave Equation")

    u, alpha = init_simulation()
    alpha_mem = cuda.to_device(alpha)
    cmap_mem = cuda.to_device(cmap)
    deriv_coffs_spc_mem = cuda.to_device(deriv_coffs_space)
    deriv_coffs_tim_mem = cuda.to_device(deriv_coffs_time)
    pixeldata = np.zeros((u.shape[1] - 2, u.shape[2] - 2, 3), dtype=np.uint8)

    while True:
        u = np.append(np.zeros((1, u.shape[1], u.shape[2]), dtype=np.float64), u[:-1], axis=0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # place_raindrops(u)
        debug = cuda.device_array((pixeldata.shape[0], pixeldata.shape[1]), dtype=np.float64)

        u = cuda.to_device(u)
        pixeldata = cuda.to_device(pixeldata)

        block_grid_1 = (int(np.ceil(u.shape[1] / thread_block[0])), int(np.ceil(u.shape[2] / thread_block[1])))
        block_grid_2 = (int(np.ceil(pixeldata.shape[0] / thread_block[0])),
                        int(np.ceil(pixeldata.shape[1] / thread_block[1])))

        generate_new_frame[block_grid_1, thread_block](u, alpha_mem, deriv_coffs_spc_mem, deriv_coffs_tim_mem)

        u = u.copy_to_host()

        # print_frame[block_grid_2, thread_block](cuda.to_device(u), pixeldata, cmap,
        #                                         max(np.abs(u.max()), np.abs(u.min())), debug)

        use_color_map[block_grid_1, thread_block](cuda.to_device(u), max(np.abs(u.max()), np.abs(u.min())),
                                                  pixeldata, cmap_mem)
        pixeldata = pixeldata.copy_to_host()

        # pixeldata[1:dimx, 1:dimy, 0] = np.clip(u[0, 1:dimx, 1:dimy] + 128, -255, 255)
        # pixeldata[1:dimx, 1:dimy, 1] = np.clip(u[1, 1:dimx, 1:dimy] + 128, -255, 255)
        # pixeldata[1:dimx, 1:dimy, 2] = np.clip(u[2, 1:dimx, 1:dimy] + 128, -255, 255)

        surf = pygame.surfarray.make_surface(pixeldata)
        display.blit(pygame.transform.scale(surf, (dimx * cellsize, dimy * cellsize)), (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    main()
