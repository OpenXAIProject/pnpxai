
def project_onto_LPIPS_ball(dist, x_mod_, x_0, eps, n=10):
    """ Projection using the Bisection Method as in https://arxiv.org/pdf/2006.12655.pdf

    Parameters
    ----------
    dist
    x_mod_
    x_0
    eps
    n

    Returns
    -------

    """
    alpha_min, alpha_max = 0, 1
    delta = x_mod_ - x_0
    for i in range(n):
        alpha = (alpha_min + alpha_max) / 2
        x_mod_temp = x_0 + alpha*delta
        if dist(x_0, x_mod_temp) > eps:
            alpha_max = alpha
        else:
            alpha_min = alpha
    return x_mod_temp