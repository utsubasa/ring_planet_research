import numpy as np
import c_compile_ring

# f


# Ring model 
# Input "x" (1d array), "pdic" (dic)
# Ouput flux (1d array)
def model(x, pdic):
        q1, q2, t0, porb, rp_rs, a_rs, b, norm \
                = pdic['q1'], pdic['q2'], pdic['t0'], pdic['porb'], pdic['rp_rs'], pdic['a_rs'], pdic['b'], pdic['norm']
        theta, phi, tau, r_in, r_out \
                = pdic['theta'], pdic['phi'], pdic['tau'], pdic['r_in'], pdic['r_out']
        norm2, norm3 = pdic['norm2'], pdic['norm3']
        cosi = b/a_rs
        u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]
        u1, u2 = u[0], u[1]
        ecosw = pdic['ecosw']
        esinw = pdic['esinw']
        """ when e & w used: ecosw -> e, esinw -> w (deg)
        e = ecosw
        w = esinw*np.pi/180.0
        ecosw, esinw = e*np.cos(w), e*np.sin(w)
        """

        # see params_def.h
        pars = np.array([porb, t0, ecosw, esinw, b, a_rs, theta, phi, tau, r_in, r_out, rp_rs, q1, q2])
        times = np.array(x)
        return np.array(c_compile_ring.getflux(times, pars, len(times)))*(
                norm + norm2*(times-t0) + norm3*(times-t0)**2)


                
