import torch
import warnings
import torch.nn.functional as F

_init_defaults = {
    'ista': 'zero',
    'cd': 'zero',
    'gpsr': 'zero',
    'iter-ridge': 'ridge',
    'interior-point': 'ridge',
    'split-bregman': 'zero',
    'own': 'zero'
}


def initialize_code(x, weight, alpha, mode):
    n_samples = x.size(0)
    n_components = weight.size(1)
    if mode == 'zero':
        z0 = x.new_zeros(n_samples, n_components)
    elif mode == 'unif':
        z0 = x.new(n_samples, n_components).uniform_(-0.1, 0.1)
    elif mode == 'lstsq':
        z0 = lstsq(x.T, weight).T
    elif mode == 'ridge':
        z0 = ridge(x.T, weight, alpha=alpha).T
    elif mode == 'transpose':
        z0 = torch.matmul(x, weight)
    else:
        raise ValueError("invalid init parameter '{}'.".format(mode))

    return z0


def sparse_encode(x, weight, alpha=1.0, z0=None, algorithm='ista', init=None,
                  **kwargs):
    n_samples = x.size(0)
    n_components = weight.size(1)

    # initialize code variable
    if z0 is not None:
        assert z0.shape == (n_samples, n_components)
    else:
        if init is None:
            init = _init_defaults.get(algorithm, 'zero')
        elif init == 'zero' and algorithm == 'iter-ridge':
            warnings.warn("Iterative Ridge should not be zero-initialized.")
        z0 = initialize_code(x, weight, alpha, mode=init)

    # perform inference
    if algorithm == 'cd':
        z = coord_descent(x, weight, z0, alpha, **kwargs)
    elif algorithm == 'gpsr':
        A = lambda v: torch.mm(v, weight.T)
        AT = lambda v: torch.mm(v, weight)
        z = gpsr_basic(x, A, tau=alpha, AT=AT, x0=z0, **kwargs)
    elif algorithm == 'iter-ridge':
        z = iterative_ridge(z0, x, weight, alpha, **kwargs)
    elif algorithm == 'ista':
        z = ista(x, z0, weight, alpha, **kwargs)
    elif algorithm == 'interior-point':
        z, _ = interior_point(x, weight, z0, alpha, **kwargs)
    elif algorithm == 'split-bregman':
        z, _ = split_bregman(weight, x, z0, alpha, **kwargs)
    elif algorithm == 'own':
        z = orthant_wise_newton(weight, x, z0, alpha, **kwargs)
    else:
        raise ValueError("invalid algorithm parameter '{}'.".format(algorithm))

    return z


def split_bregman(A, y, x0=None, alpha=1.0, lambd=1.0, maxiter=20, niter_inner=5,
                  tol=1e-10, tau=1., TV=False, beta=2.0, verbose=False):
    """Split Bregman for L1-regularized least squares.

    Parameters
    ----------
    A : torch.Tensor
        Linear transformation matrix. Shape [n_features, n_components]
    y : torch.Tensor
        Reconstruction targets. Shape [n_samples, n_features]
    x0 : torch.Tensor, optional
        Initial guess at the solution. Shape [n_samples, n_components]
    alpha : float
        L1 Regularization strength
    lambd : float
        Dampening term; constraint penalty strength
    maxiter : int
        Number of iterations of outer loop
    niter_inner : int
        Number of iterations of inner loop
    tol : float, optional
        Tolerance on change in parameter x
    tau : float, optional
        Scaling factor in the Bregman update (must be close to 1)
    TV : bool, optional
        Whether to apply TV regularization
    beta : float, optional
        TV regularization strength
    verbose : bool, optional
        Whether to print iteration information

    Returns
    -------
    x : torch.Tensor
        Sparse coefficients. Shape [n_samples, n_components]
    itn_out : int
        Iteration number of outer loop upon termination

    """
    assert y.dim() == 2
    assert A.dim() == 2
    assert y.shape[1] == A.shape[0]
    n_features, n_components = A.shape
    n_samples = y.shape[0]
    y = y.T.contiguous()
    if x0 is None:
        x = y.new_zeros(n_components, n_samples)
    else:
        assert x0.shape == (n_samples, n_components)
        x = x0.T.clone(memory_format=torch.contiguous_format)

    # sb buffers
    b = torch.zeros_like(x)
    d = torch.zeros_like(x)

    # normal equations
    Aty = torch.mm(A.T, y) / alpha
    AtA = torch.mm(A.T, A) / alpha
    AtA.diagonal(dim1=-2, dim2=-1).add_(lambd)
    AtA_inv = torch.cholesky_inverse(torch.linalg.cholesky(AtA))

    update = y.new_tensor(float('inf'))
    for itn in range(maxiter):
        if update <= tol:
            break

        xold = x.clone()
        for _ in range(niter_inner):
            # Regularized sub-problem
            Aty_i = Aty.add(d - b, alpha=lambd)
            x = torch.mm(AtA_inv, Aty_i)

            if TV:
                d = tv_regularization(x, beta, lambd)
            else:
                # Shrinkage
                d = F.softshrink(x + b, 1 / lambd)

        # Bregman update
        b.add_(x - d, alpha=tau)

        # update norm
        update = torch.norm(x - xold)

        if verbose:
            cost = 0.5 * (torch.mm(A, x) - y).square().sum() + alpha * x.abs().sum()
            print('iter %3d - cost: %0.4f' % (itn, cost))

    x = x.T.contiguous()

    return x, itn



def tv_regularization(x, beta, lambd):
    """Apply Total Variation (TV) regularization."""
    # Reshape x back to a 2D image
    x_2d = x.reshape(32, 32)

    # Compute gradients along x and y axes
    grad_x = F.conv2d(x_2d.unsqueeze(0).unsqueeze(0), torch.Tensor([[[[1, -1]]]]), padding=(0, 1))
    grad_y = F.conv2d(x_2d.unsqueeze(0).unsqueeze(0), torch.Tensor([[[[1], [-1]]]]), padding=(1, 0))  # Adjusted padding

    # Crop the output tensors to match the size of the input image
    grad_x = grad_x[:, :, :, :-1]  # Remove the last column
    grad_y = grad_y[:, :, :-1, :]  # Remove the last row

    # Compute magnitude of gradients
    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    # Apply soft shrinkage to gradients
    v = F.softshrink(grad, beta / lambd)

    # Compute divergence
    div_v_x = F.conv2d(v, torch.Tensor([[[[1], [-1]]]]), padding=(1, 0))
    div_v_y = F.conv2d(v, torch.Tensor([[[[1, -1]]]]), padding=(0, 1))

    # Crop the output tensors to match the size of the input image
    div_v_y = div_v_y[:, :, :, :-1]  # Remove the last column
    div_v_x = div_v_x[:, :, :-1, :]  # Remove the last row

    div_v = div_v_x + div_v_y

    # Add divergence to original image
    x_2d_reg = x_2d + div_v.squeeze()

    # Reshape back to 1D array
    x_reg = x_2d_reg.view(1024, 1)

    return x_reg
