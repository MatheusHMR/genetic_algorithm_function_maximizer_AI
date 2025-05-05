import numpy as np
from scipy.optimize import basinhopping, minimize, Bounds

# Função negativa para maximizar
def f_obj(x):
    x1, x2 = x
    return -(21.5 + x1*np.sin(4*np.pi*x1) + x2*np.sin(20*np.pi*x2))

# Limites para x1 e x2
bounds = Bounds([-3.1, 4.1], [12.1, 5.8])

# Minimizer kwargs usando L-BFGS-B com bounds
minimizer_kwargs = {
    "method": "L-BFGS-B",
    "bounds": bounds
}

# Ponto inicial
x0 = np.array([0.0, 5.0])

ret = basinhopping(f_obj, x0,
                   minimizer_kwargs=minimizer_kwargs,
                   niter=20000,
                   stepsize=0.01)
x_opt, f_max = ret.x, -ret.fun
print(f"Máximo em x1={x_opt[0]:.6f}, x2={x_opt[1]:.6f} → f={f_max:.6f}")
