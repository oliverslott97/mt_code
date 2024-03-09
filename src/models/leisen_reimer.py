import numpy as np
from scipy.optimize import fsolve
import scipy as sq

def LeisenReimerBinomial(AmeEurFlag, CallPutFlag, AdjFlag, S, X, T, r, c, v, n, t_vec, d_vec):
    if AdjFlag == True:
        v = np.sqrt(np.sum(((S*v)/(S - d_vec))**2*(np.insert(t_vec, 0, 0)))+v**2*(T - t_vec[-1]))
    else:
        v = v
    
    n = round_up_to_odd(n)
    n_list = np.arange(0, (n + 1), 1)

    if CallPutFlag == 'C':
        z = 1
    elif CallPutFlag == 'P':
        z = -1
    else:
        return 'Call or put not defined'

    d1 = (np.log(S / X) + (c + v ** 2 / 2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)

    hd1 = 0.5 + np.sign(d1) * (0.25 - 0.25 * np.exp(-(d1 / (n + 1 / 3 + 0.1 / (n + 1))) ** 2 * (n + 1 / 6))) ** 0.5
    hd2 = 0.5 + np.sign(d2) * (0.25 - 0.25 * np.exp(-(d2 / (n + 1 / 3 + 0.1 / (n + 1))) ** 2 * (n + 1 / 6))) ** 0.5

    dt = T / n
    p = hd2

    u = np.exp(c * dt) * hd1 / hd2
    d = (np.exp(c * dt) - p * u) / (1 - p)
    df = np.exp(-r * dt)

    max_pay_off_list = []
    for i in n_list:
        i = i.astype('int')
        max_pay_off = np.maximum(0, z * (S * u ** i * d ** (n - i) - X))
        max_pay_off_list.append(max_pay_off)

    for j in np.arange(n - 1, 0 - 1, -1):
        for i in np.arange(0, j + 1, 1):
            i = i.astype(int)
            if AmeEurFlag == 'e':
                max_pay_off_list[i] = (p * max_pay_off_list[i + 1] + (1 - p) * max_pay_off_list[i]) * df
            elif AmeEurFlag == 'a':
                max_pay_off_list[i] = np.maximum((z * (S * u ** i * d ** (j - i) - X)),
                                                 (p * max_pay_off_list[i + 1] + (1 - p) * max_pay_off_list[i]) * df)
    price = max_pay_off_list[0]

    return price


def round_up_to_odd(n):
    return np.ceil(n) // 2 * 2 + 1


def IV_solver(AmeEurFlag, CallPutFlag, AdjFlag, S, X, T, r, c, n, Option_Value, t_vec=None, d_vec=None):
    def objection_function(IV):
        result = Option_Value - LeisenReimerBinomial(AmeEurFlag, CallPutFlag, AdjFlag, S, X, T, r, c, IV, n, t_vec, d_vec)
        
        return result

    IV_Result = sq.optimize.brentq(objection_function, a=0.1, b=2, xtol=0.000001)
    
    return IV_Result