import matplotlib.pyplot as plt
import numpy as np
import math

#i)
p_3x8 = (1/8) ** 3 * 8 # 1 / 64
p_4x5 = (1/5) ** 4 * 5 # 1 / 125

print("i)")
print(f"3x8: P(X) = {p_3x8}")
print(f"4x5: P(X) = {p_4x5}")

#ii)
p_3x8_diff = math.perm(8, 3) * p_3x8
p_4x5_diff = math.perm(5, 4) * p_4x5
print("ii)")
print(f"3x8: P(X) = {p_3x8_diff}")
print(f"4x5: P(X) = {p_4x5_diff}")

# iii)
n = 5
x = 2

x_i = np.arange(0, n+1)

combinations = [math.comb(n, x_i[k]) for k in range(len(x_i))]
f_x_3x8 = [combinations[k] * (p_3x8 ** x_i[k]) * ((1 - p_3x8) ** (n - x_i[k])) for k in range(len(x_i))]
f_x_4x5 = [combinations[k] * (p_4x5 ** x_i[k]) * ((1 - p_4x5) ** (n - x_i[k])) for k in range(len(x_i))]


p_x_beq_2_3x8 = np.sum(f_x_3x8[x:n])
p_x_beq_2_4x5 = np.sum(f_x_4x5[x:n])

mu_3x8 = n * p_3x8
mu_4x5 = n * p_4x5

var_3x8 = n * p_3x8 * (1 - p_3x8)
var_4x5 = n * p_4x5 * (1 - p_4x5)


print("iii)")
print("3x8:")
print(f"- P(X >= 2) = {p_x_beq_2_3x8}")
print(f"- Moyenne = {mu_3x8}")
print(f"- Variance = {var_3x8}")
print("4x5:")
print(f"- P(X >= 2) = {p_x_beq_2_4x5}")
print(f"- Moyenne = {mu_4x5}")
print(f"- Variance = {var_4x5}")