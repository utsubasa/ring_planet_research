import scipy.stats
from scipy import integrate
from scipy.optimize import minimize_scalar


def p_value_function(F_obs):
    p_value = (
        1
        - integrate.quad(
            lambda x: scipy.stats.f.pdf(
                x,
                500 - 9 - 1,
                9 - 6,
            ),
            0,
            F_obs,
        )[0]
    )
    return p_value


print(p_value_function(65))
"""
result = minimize_scalar(p_value_function)
optimal_F_obs = result.x
optimal_p_value = p_value_function(optimal_F_obs)

print("Optimal F_obs:", optimal_F_obs)
print("Minimum p-value:", optimal_p_value)
"""
