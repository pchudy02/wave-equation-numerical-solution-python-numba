import copy

import pygame
import numpy as np
import random
from scipy.signal import savgol_coeffs
from numba import cuda, float64, int32
import matplotlib as mpl
import multiprocessing as mp


def deriv_matrix_generator(window):
    def get_area(x, y):
        integral = lambda x, y: (0.5 * x * np.sqrt((y ** 2) + y - (x ** 2) + 0.25) +
                                 0.5 * (y + 0.5) ** 2 * np.arctan(x / np.sqrt(y ** 2 + y - x ** 2 + 0.25)) -
                                 y * x + x * 0.5)

        if x == 0 and y == 0:
            return np.pi * 0.5 ** 2

        elif x == y:
            return integral(np.sqrt((x + 0.5) ** 2) - 0.25, y) - integral(x - 0.5, y)
        else:

            return integral(x + 0.5, y) - integral(x - 0.5, y)

    ring_area = lambda r: np.pi * ((r + 0.5) ** 2 - (r - 0.5) ** 2) if r > 0 else np.pi * (0.5 ** 2)

    deriv_coffs_space = np.append(savgol_coeffs(window * 2 + 1, window * 2, deriv=2, use='dot')[window:], 0)

    deriv_matrix = np.zeros([window * 2 + 1] * 2, dtype=np.float64)

    for y in range(1, window + 1):
        for x in range(y + 1):
            if x == 0:
                r = np.uint16(y)
            else:
                r = np.uint16(np.round(np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)))

            if r > window:
                continue

            part = get_area(x, y)
            deriv_matrix[x + window, y + window] = part * deriv_coffs_space[r] / ring_area(r)

            if r < window:
                deriv_matrix[x + window, y + window] += (1 - part) * deriv_coffs_space[r + 1] / ring_area(r + 1)

            deriv_matrix[y + window, x + window] = deriv_matrix[x + window, y + window]

    for x in range(window + 1, window * 2 + 1):
        deriv_matrix[window * 2 - x, :] = deriv_matrix[x, :]

    for x in range(window + 1, window * 2 + 1):
        deriv_matrix[:, window * 2 - x] = deriv_matrix[:, x]

    deriv_matrix[window, window] = -deriv_matrix.sum()

    result = np.array(deriv_matrix[window:, window:])

    return result


h = 1  # spatial step width
t = 1  # time step width
dimx = 1000  # width of the simulation domain
dimy = 1000  # height of the simulation domain
deriv_window = 8
widow_size = (1000, 1000)
deriv_coffs_space = deriv_matrix_generator(deriv_window)
deriv_coffs_time = np.array(savgol_coeffs(3, 2, deriv=2, use='dot'), dtype=np.float64)
cmap = np.array([np.array(mpl.cm.inferno(i)[:3]) * 256 for i in np.linspace(0, 1, 256)], dtype=np.uint8)
pass
thread_block = (32, 32)


def init_simulation():
    u = np.zeros((3, dimx + 2, dimy + 2), dtype=np.float64)
    u[0, u.shape[1] // 2:u.shape[1] // 2 + 2, u.shape[2] // 2:u.shape[2] // 2 + 2] = 1000
    # u[0, 1:4, 1:4] = 100
    # u[0, dimx // 2, :] = 1000
    # u[0, 9, 9] = 10
    # The three dimensional simulation grid u[time, dimx, dimy]

    c = 0.5  # The "original" wave propagation speed

    alpha = np.zeros((u.shape[1], u.shape[2]), dtype=np.float64)
    # wave propagation velocities of the entire simulation domain

    alpha[:, :] = ((c * t) / h) ** 2  # will be set to a constant value of tau

    return u, alpha


# @cuda.jit(f'void(float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:])')
# def update(u, result, alpha, deriv_coffs):
#     x, y = cuda.grid(2)
#     idx = cuda.threadIdx.x
#     idy = cuda.threadIdx.y
#
#     deriv_coffs_loc = cuda.shared.array(shape=5, dtype=np.float64)
#
#     # shape=(offset * 2 + 1, offset * 2 + 32, offset * 2 + 32)
#     u_loc = cuda.shared.array(shape=(5, 12, 8), dtype=np.float64)
#     alpha_loc = cuda.shared.array(shape=(8, 4), dtype=np.float64)
#
#     if x < u.shape[1] and y < u.shape[2]:
#
#         i = 0
#         while i + idx * 4 + idy < deriv_coffs.shape[0]:
#             deriv_coffs_loc[i + idx * 4 + idy] = deriv_coffs[i + idx * 4 + idy]
#             i += 32  # cuda.blockDim[0] * cuda.blockDim[1]
#
#         if 0 < x < u.shape[1] - 1 and 0 < y < u.shape[2] - 1:
#             alpha_loc[idx, idy] = alpha[x, y]
#             # result[0, x, y] = alpha_loc[idx, idy]
#             # alpha_loc[idx, idy] = x + y
#
#         i, j = -deriv_window - 4, -deriv_window - 2
#         while i + idx < 4 + deriv_window:
#             while j + idy < 2 + deriv_window:
#                 if 0 <= i + x < u.shape[1] and 0 <= j + y < u.shape[2]:
#                     k = 1
#                     while k < u.shape[0]:
#                         u_loc[k, i + idx, j + idy] = u[k - 1, i + x, j + y]
#                         k += 1
#                     pass
#
#                 j += 4
#             i += 8
#
#         cuda.syncthreads()
#         # u_loc[0, idx, idy] = alpha_loc[idx, idx]
#         result[0, x, y] = u_loc[0, idx + deriv_window, idy + deriv_window]
#
#         """With multiplying after second loop throws: {CudaAPIError}[700] Call to cuMemcpyDtoH results in
#         UNKNOWN_CUDA_ERROR"""
#
#         if 0 < x < u.shape[1] - 1 and 0 < y < u.shape[2] - 1:
#             u_loc[0, idx + deriv_window, idy + deriv_window] = alpha_loc[idx, idx]
#             # result[0, x, y] = u_loc[0, idx + offset, idy + offset]
#             # u[0, x, y] = alpha_loc[idx, idx]
#             # cuda.syncwarp()
#
#             temp = 0
#             i = 0
#             while i < deriv_window * 2:
#                 if 0 < i + x - deriv_window < u.shape[1] - 1 and 0 <= i + idx - deriv_window < 8:
#                     temp += (u_loc[int(u.shape[0] // 2), i + idx - deriv_window, idy] * deriv_coffs_loc[i])
#                     pass
#                 i += 1
#
#             # result[0, x, y] = alpha_loc[idx, idx]
#
#             # i = 0
#             # while i < offset * 2:
#             #     if 0 < i + y - offset < u.shape[1] - 1:
#             #         temp += u_loc[int(u.shape[0] // 2), x, i + y - offset] * deriv_coffs_loc[i]
#             #     i += 1
#             #
#             # result[0, x, y] = alpha_loc[idx, idx]
#             # This multiplying
#             # u_loc[0, idx, idy] = u_loc[0, idx, idy] * temp
#             #
#             # i = 1
#             # while i < offset * 2:
#             #     u_loc[0, idx, idy] += u_loc[i, x, y] * (-deriv_coffs_loc[i])
#             #     i += 1
#             #
#             # result[0, x, y] = u_loc[0, idx, idy] / deriv_coffs_loc[0] * 0.995
#
#         # elif (x == 0 or x == u.shape[1] - 1) ^ (y == 0 or y == u.shape[2] - 1):
#         #     pass


@cuda.jit(f'void(float64[:,:,:], float64[:,:], float64[:,:], float64[:])')
def generate_new_frame(u, alpha, deriv_coffs_space, deriv_coffs_time):
    x, y = cuda.grid(2)
    idx = cuda.threadIdx.x
    idy = cuda.threadIdx.y

    u_shared = cuda.shared.array((thread_block[0] + deriv_window * 2, thread_block[1] + deriv_window * 2),
                                 dtype=np.float64)
    a_shared = cuda.shared.array((thread_block[0] + deriv_window * 2, thread_block[1] + deriv_window * 2),
                                 dtype=np.float64)

    if x < u.shape[1] and y < u.shape[2]:
        for i in range(thread_block[0] + deriv_window * 2, __step=32):
            for j in range(thread_block[1] + deriv_window * 2, __step=32):
                if 0 <= x - deriv_window + i <= u.shape[1] and 0 <= y - deriv_window + j <= u.shape[2]:
                    u_shared[1, idx + i, idy + j] = u[0, x - deriv_window + i, y - deriv_window + j]
                    u_shared[2, idx + i, idy + j] = u[1, x - deriv_window + i, y - deriv_window + j]
                    a_shared[idx + i, idy + j] = alpha[x - deriv_window + i, y - deriv_window + j]

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
        # $$$ Generacja następnej klatki $$$ #
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #

        if 0 < x < u.shape[1] - 1 and 0 < y < u.shape[2] - 1:
            # x_plus = True
            # x_minus = True
            # y_plus = True
            # y_minus = True
            #
            # x_plus_y_plus = True
            # x_plus_y_minus = True
            # x_minus_y_plus = True
            # x_minus_y_minus = True

            temp = 0
            for i in range(deriv_window + 1):

                # Pion - poziom #

                # if u.shape[1] - 1 > x + i > 0 and x_plus:
                #     if alpha[x + i, y] == 0:
                #         x_plus = False
                #     else:
                #         temp += u[1, x + i, y] * deriv_coffs_space[i]
                #
                # if u.shape[1] - 1 > x - i > 0 != i and x_minus:
                #     if alpha[x - i, y] == 0:
                #         x_minus = False
                #     else:
                #         temp += u[1, x - i, y] * deriv_coffs_space[i]
                #
                # if u.shape[1] - 1 > y + i > 0 and y_plus:
                #     if alpha[x, y + i] == 0:
                #         y_plus = False
                #     else:
                #         temp += u[1, x, y + i] * deriv_coffs_space[i]
                #
                # if u.shape[1] - 1 > y - i > 0 != i and y_minus:
                #     if alpha[x, y - i] == 0:
                #         y_minus = False
                #     else:
                #         temp += u[1, x, y - i] * deriv_coffs_space[i]
                #
                # # Ukosy #
                #
                # if u.shape[1] - 1 > x + i > 0 and u.shape[2] - 1 > y + i > 0 and x_plus_y_plus:
                #     if alpha[x + i, y + i] == 0:
                #         x_plus = False
                #     else:
                #         temp += u[1, x + i, y + i] * deriv_coffs_space[i]
                #
                # if u.shape[1] - 1 > x - i > 0 != i and u.shape[2] - 1 > y - i > 0 and x_minus_y_minus:
                #     if alpha[x - i, y - i] == 0:
                #         x_plus = False
                #     else:
                #         temp += u[1, x - i, y - i] * deriv_coffs_space[i]
                #
                # if u.shape[1] - 1 > x - i > 0 and u.shape[2] - 1 > y + i > 0 and x_minus_y_plus:
                #     if alpha[x - i, y + i] == 0:
                #         x_plus = False
                #     else:
                #         temp += u[1, x - i, y + i] * deriv_coffs_space[i]
                #
                # if u.shape[1] - 1 > x + i > 0 != i and u.shape[2] - 1 > y - i > 0 and x_plus_y_minus:
                #     if alpha[x + i, y - i] == 0:
                #         x_plus = False
                #     else:
                #         temp += u[1, x + i, y - i] * deriv_coffs_space[i]

                # temp *= (alpha[x, y] ** 2)

                # Pochodne w 2d (3d)
                for j in range(deriv_window + 1):
                    if u.shape[1] - 1 > x + i > 0 and u.shape[2] - 1 > y + j > 0:
                        temp += u[1, x + i, y + j] * deriv_coffs_space[i, j] * alpha[x + i, y + j]

                    if u.shape[1] - 1 > x + i > 0 and u.shape[2] - 1 > y - j > 0 < j and not (i == 0 and j == 0):
                        temp += u[1, x + i, y - j] * deriv_coffs_space[i, j] * alpha[x + i, y - j]

                    if u.shape[1] - 1 > x - i > 0 < i and u.shape[2] - 1 > y + j > 0 and not (i == 0 and j == 0):
                        temp += u[1, x - i, y + j] * deriv_coffs_space[i, j] * alpha[x - i, y + j]

                    if u.shape[1] - 1 > x - i > 0 < i and u.shape[2] - 1 > y - j > 0 < j and not (i == 0 and j == 0):
                        temp += u[1, x - i, y - j] * deriv_coffs_space[i, j] * alpha[x - i, y - j]

            i = 1
            while i < deriv_coffs_time.shape[0]:
                temp -= u[i, x, y] * deriv_coffs_time[i]
                i += 1

            if (10 ** (-310)) >= temp / deriv_coffs_time[0] >= -(10 ** (-310)):
                u[0, x, y] = 0
                # u_shared[idx, idy] = 0
            else:
                u[0, x, y] = (temp / deriv_coffs_time[0])
                # u_shared[idx, idy] = (temp / deriv_coffs_time[0])

        elif (x == 0 or x == u.shape[1] - 1) ^ (y == 0 or y == u.shape[2] - 1):
            # k = alpha[x, y] * t / h
            # k = (k - 1) / (k + 1)
            # if x == 0:
            #     u[0, x, y] = u[1, x + 1, y] + k * (u[0, x + 1, y] - u[1, x, y])
            # elif x == u.shape[1] - 1:
            #     u[0, x, y] = u[1, x - 1, y] + k * (u[0, x - 1, y] - u[1, x, y])
            # elif y == 0:
            #     u[0, x, y] = u[1, x, y + 1] + k * (u[0, x, y + 1] - u[1, x, y])
            # elif y == u.shape[1] - 1:
            #     u[0, x, y] = u[1, x, y - 1] + k * (u[0, x, y - 1] - u[1, x, y])
            pass


def max_abs_value_from_array(arr):
    sector_size = (2, 2)

    a_mem = cuda.device_array((dimx + 2, dimy + 2), dtype=np.float64)
    b_mem = cuda.device_array((int(np.ceil((dimx + 2) / sector_size[0])),
                               int(np.ceil((dimy + 2) / sector_size[1]))),
                              dtype=np.float64)

    @cuda.jit()
    def copy_array(u, a):
        x, y = cuda.grid(2)

        if x < u.shape[1] and y < u.shape[2]:
            a[x, y] = u[0, x, y]

        cuda.syncthreads()

    @cuda.jit('void(float64[:,:], float64[:,:])')
    def max_val_from_arr_gpu(a, b):
        x, y = cuda.grid(2)

        if x < b.shape[0] and y < b.shape[1]:
            temp = 0

            for i in range(2):
                for j in range(2):
                    if x * 2 + i < a.shape[0] and y * 2 + j < a.shape[1]:
                        temp1 = a[x * 2 + i, y * 2 + j]
                        if temp < temp1:
                            temp = temp1

            b[x, y] = temp

        cuda.syncthreads()

    copy_array[(int(np.ceil((dimx + 2) / 32)), int(np.ceil((dimy + 2) / 32))), (32, 32)](arr, a_mem)

    block_grid = (int(np.ceil((dimx + 2) / sector_size[0] / 32)), int(np.ceil((dimy + 2) / sector_size[1] / 32)))

    for k in range(int(np.ceil(max(np.log2(arr.shape[1]) / np.log2(sector_size[0]),
                                   np.log2(arr.shape[2]) / np.log2(sector_size[1]))))):
        max_val_from_arr_gpu[block_grid, (32, 32)](a_mem, b_mem)
        a_mem, b_mem = b_mem, a_mem
        block_grid = (int(np.ceil(block_grid[0] / sector_size[0])), int(np.ceil(block_grid[1] / sector_size[1])))

    return (a_mem.copy_to_host())[0, 0]


@cuda.jit('void(float64[:,:,:], uint8[:,:,:], uint8[:,:], float64)')
def print_frame(u, pixeldata, cmap, max_val_data):
    x, y = cuda.grid(2)

    if x < pixeldata.shape[0] and y < pixeldata.shape[1]:
        x_low = x * (u.shape[1] - 2) / pixeldata.shape[0]
        x_high = (x + 1) * (u.shape[1] - 2) / pixeldata.shape[0]
        y_low = y * (u.shape[2] - 2) / pixeldata.shape[1]
        y_high = (y + 1) * (u.shape[2] - 2) / pixeldata.shape[1]

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

                pixel_value += jx * jy * u[0, int(ix // 1) + 1, int(iy // 1) + 1]

                iy += jy
            ix += jx

        pixel_value /= ((u.shape[1] - 2) / pixeldata.shape[0]) * ((u.shape[2] - 2) / pixeldata.shape[1])

        i = 0
        c = (int(pixel_value / max_val_data * 127)) + 127

        if c < 0:
            c = 0
        elif c > 255:
            c = 255

        while i < 3:
            if max_val_data != 0:
                pixeldata[x, y, i] = cmap[c, i]

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


def place_raindrops(u, prob):
    if (random.random() < prob) and True:
        u[0, random.randrange(1, dimx + 1), random.randrange(1, dimx + 1)] = np.random.randint(10, 10000)


def backend(pipe, queue):
    u, alpha = init_simulation()
    alpha_mem = cuda.to_device(alpha)
    cmap_mem = cuda.to_device(cmap)
    deriv_coffs_spc_mem = cuda.to_device(deriv_coffs_space)
    deriv_coffs_tim_mem = cuda.to_device(deriv_coffs_time)
    pixeldata = cuda.device_array((*widow_size, 3), dtype=np.uint8)

    while True:
        if not queue.full():
            u = np.append(np.zeros((1, u.shape[1], u.shape[2]), dtype=np.float64), u[:-1], axis=0)
            u_mem = cuda.to_device(u)

            block_grid_1 = (int(np.ceil(u.shape[1] / thread_block[0])), int(np.ceil(u.shape[2] / thread_block[1])))
            block_grid_2 = (int(np.ceil(pixeldata.shape[0] / thread_block[0])),
                            int(np.ceil(pixeldata.shape[1] / thread_block[1])))

            generate_new_frame[block_grid_1, thread_block](u_mem, alpha_mem, deriv_coffs_spc_mem, deriv_coffs_tim_mem)

            u = u_mem.copy_to_host()

            print_frame[block_grid_2, thread_block](u_mem, pixeldata, cmap_mem, np.abs(u).max())

            queue.put(pixeldata.copy_to_host())


def main1():
    pygame.init()
    display = pygame.display.set_mode(widow_size)
    pygame.display.set_caption("Solving the 2d Wave Equation")

    u, alpha = init_simulation()
    alpha_mem = cuda.to_device(alpha)
    cmap_mem = cuda.to_device(cmap)
    deriv_coffs_spc_mem = cuda.to_device(deriv_coffs_space)
    deriv_coffs_tim_mem = cuda.to_device(deriv_coffs_time)

    u0_sum = []

    while True:
        # place_raindrops(u, 0.01)

        u = np.append(np.zeros((1, u.shape[1], u.shape[2]), dtype=np.float64), u[:-1], axis=0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        u = cuda.to_device(u)

        pixeldata = cuda.device_array((*widow_size, 3), dtype=np.uint8)

        block_grid_1 = (int(np.ceil(u.shape[1] / thread_block[0])), int(np.ceil(u.shape[2] / thread_block[1])))
        block_grid_2 = (int(np.ceil(pixeldata.shape[0] / thread_block[0])),
                        int(np.ceil(pixeldata.shape[1] / thread_block[1])))

        generate_new_frame[block_grid_1, thread_block](u, alpha_mem, deriv_coffs_spc_mem, deriv_coffs_tim_mem)

        u = u.copy_to_host()
        # u0_sum.append((np.abs(u[0]).sum(), np.abs(u[0]).sum() / np.abs(u[1]).sum()))

        print_frame[block_grid_2, thread_block](cuda.to_device(u), pixeldata, cmap_mem,
                                                max(np.abs(u.max()), np.abs(u.min())))

        # print_frame[block_grid_2, thread_block](u, pixeldata, cmap, 1000)

        pixeldata = pixeldata.copy_to_host()

        surf = pygame.surfarray.make_surface(pixeldata)
        display.blit(pygame.transform.scale(surf, widow_size), (0, 0))
        pygame.display.update()


def main():
    pygame.init()
    display = pygame.display.set_mode(widow_size)
    pygame.display.set_caption("Solving the 2d Wave Equation")

    q = mp.Queue(5)
    p = mp.Pipe()

    backend_proc = mp.Process(target=backend, args=(p, q,))

    backend_proc.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                backend_proc.join()
                return

        if not q.empty():
            surf = pygame.surfarray.make_surface(q.get())
            display.blit(pygame.transform.scale(surf, widow_size), (0, 0))
            pygame.display.update()

    pass


if __name__ == "__main__":
    main()
