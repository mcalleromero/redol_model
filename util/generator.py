import numpy as np
import numpy.matlib

sys.path.append('/home/mario.calle/master/redol_model/')


def create_full_dataset(n, dimm, model, noise=None):
    x = np.random.rand(n, dimm) * 2.0 * np.pi

    if model == 'threenorm':
        a = 2 / np.sqrt(dimm)

        n2 = int(n / 2)
        n4 = int(n / 4)

        x[:n4, :] = np.random.normal(a, 1.0, (n4, dimm))
        x[n4:n2, :] = np.random.normal(-a, 1.0, (n4, dimm))
        x[n2:, :] = np.random.normal(a, 1.0, (n2, dimm))
        x[n2:, 1::2] = -x[n2:, 1::2]

        c = np.ones((n, 1))
        c[:n2, :] = 0

    elif model == 'twonorm':
        a = 2 / np.sqrt(dimm)
        n2 = int(n / 2)
        x[:n2, :] = np.random.normal(a, 1.0, (n2, dimm))
        x[n2:, :] = np.random.normal(-a, 1.0, (n2, dimm))

        c = np.ones((n, 1))
        c[:n2, :] = 0

    elif model == "ringnorm":
        a = dimm / np.sqrt(dimm)

        n2 = int(n / 2)

        x[:n2, :] = np.random.normal(0, 4.0, (n2, dimm))
        x[n2:, :] = np.random.normal(a, 1.0, (n2, dimm))

        c = np.ones((n, 1))
        c[:n2, :] = 0

    elif model == "ringnorm_normal":
        a = 2 / np.sqrt(dimm)

        n2 = int(n / 2)

        # tiene que dar 1% de error approx.
        # multivariate_normal
        x[:n2, :] = np.random.multivariate_normal(
            np.repeat(0, dimm), 4.0*np.eye(dimm), n2)
        x[n2:, :] = np.random.multivariate_normal(
            np.repeat(a, dimm), np.eye(dimm), n2)

        c = np.ones((n, 1))
        c[:n2, :] = 0

    if noise is not None:
        x = x + noise * np.random.randn(n, 1)

    return x, c * 1
