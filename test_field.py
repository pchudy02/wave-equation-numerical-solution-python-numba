import numpy as np
from scipy.signal import savgol_coeffs
from multiprocessing import Process, Queue, Pipe

# def deriv_matrix_generator(window):
#     def get_area(x, y):
#         integral = lambda x, y: (0.5 * x * np.sqrt((y ** 2) + y - (x ** 2) + 0.25) +
#                                  0.5 * (y + 0.5) ** 2 * np.arctan(x / np.sqrt(y ** 2 + y - x ** 2 + 0.25)) -
#                                  y * x + x * 0.5)
#
#         if x == 0 and y == 0:
#             return np.pi * 0.5 ** 2
#
#         elif x == y:
#             return integral(np.sqrt((x + 0.5) ** 2) - 0.25, y) - integral(x - 0.5, y)
#         else:
#
#             return integral(x + 0.5, y) - integral(x - 0.5, y)
#
#     ring_area = lambda r: np.pi * ((r + 0.5) ** 2 - (r - 0.5) ** 2) if r > 0 else np.pi * (0.5 ** 2)
#
#     deriv_coffs_space = np.append(savgol_coeffs(window * 2 + 1, window * 2, deriv=2, use='dot')[window:], 0)
#
#     deriv_matrix = np.zeros([window * 2 + 1] * 2, dtype=np.float64)
#
#     for y in range(1, window + 1):
#         for x in range(y + 1):
#             if x == 0:
#                 r = np.uint16(y)
#             else:
#                 r = np.uint16(np.round(np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)))
#
#             part = get_area(x, y)
#             deriv_matrix[x + window, y + window] = part * deriv_coffs_space[r] / ring_area(r) + (1 - part) * \
#                                                    deriv_coffs_space[r + 1] / ring_area(r + 1)
#
#             deriv_matrix[y + window, x + window] = deriv_matrix[x + window, y + window]
#
#     for x in range(window + 1, window * 2 + 1):
#         deriv_matrix[window * 2 - x, :] = deriv_matrix[x, :]
#
#     for x in range(window + 1, window * 2 + 1):
#         deriv_matrix[:, window * 2 - x] = deriv_matrix[:, x]
#
#     deriv_matrix[window, window] = -deriv_matrix.sum()
#
#     return deriv_matrix[window:, window:]
#
#
# # temp = deriv_matrix_generator(2)
# # temp1 = deriv_matrix_generator(2)
#
# # i = 1
# # temp = []
# # while True:
# #     temp1 = []
# #     for j in range(2, i, 2):
# #         temp1.append(savgol_coeffs(i, j, deriv=2, use='dot')[i // 2:])
# #
# #     temp.append(temp1)
# #     i += 2
# #
# # pass
#
#
# def f(q):
#     q.send([42, None, 'hello'])
#     for i in range(4):
#         for j in range(i):
#             temp = [j]*j
#             q.send(temp)
#
#     return None
#
#
# if __name__ == '__main__':
#     parent_conn, child_conn = Pipe()
#     p = Process(target=f, args=(child_conn,))
#     p.start()
#     print(parent_conn.recv())  # prints "[42, None, 'hello']"
#     p.join()
#     pass

from scipy.interpolate import lagrange
from scipy.signal import savgol_coeffs

window = 10
y = np.append(savgol_coeffs(window * 2 + 1, window * 2, deriv=2, use='dot')[window:], [0, 0, 0, 0, 0, 0, 0, 0], axis=0)

# Podaj swoje punkty jako listy. Na przykład:
x = np.array([i for i in range(y.shape[0])])

# Użyj metody Lagrange'a do wygenerowania funkcji interpolującej
poly = lagrange(x, y)
[print(f'{poly.coef[i]}x^{y.shape[0] - i - 1} + ', end=' ') for i in range(y.shape[0])]
print()

[print(f'Jeżeli(x<={i + 0.5}, {y[i]},', end=' ') for i in range(y.shape[0])]
print()

pass
# Teraz "poly" to funkcja, którą możesz użyć do obliczenia y dla dowolnego x
