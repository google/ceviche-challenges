from javiche import jaxit
import ceviche
from ceviche_challenges.jax_scattering import calculate_amplitudes


def ceviche_simulate(excite_port_idx, omega, dl, epsilon_r_bg, pml_width, epsilon_r, ports):
    @jaxit()
    def _simulate(epsilon_r):
        sim = ceviche.fdfd_ez(
            omega,
            dl,
            epsilon_r_bg,
            [pml_width, pml_width],
        )
        sim.eps_r = epsilon_r
        source = ports[excite_port_idx].source_fdfd(
            omega,
            dl,
            epsilon_r_bg,
        )
        hx, hy, ez = sim.solve(source)

        sm = []
        sp = []
        for j, port in enumerate(ports):
            a, b = calculate_amplitudes(
                omega,
                dl,
                port,
                ez,
                hy,
                hx,
                epsilon_r_bg,
            )
            if j == excite_port_idx:
                sp = a
            sm.append(b)
        return [smi / sp for smi in sm], ez
    
    res = _simulate(epsilon_r)
    #print(res)
    return res

