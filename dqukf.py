"""
Dual Quaternion Unscented Kalman Filter (DQ-UKF).

Implements an Unscented Kalman Filter on the dual quaternion manifold for
6-DOF pose estimation. Sigma points are computed in the tangent space and
mapped back to the manifold via retract/lift operations defined in
quaternion.py, preserving unit-norm and vector-dual constraints.
"""
import numpy as np
from scipy.linalg import sqrtm
from scipy.integrate import solve_ivp
import quaternion as qn


class DQUKF:
    """
    Vectorized Unscented Kalman Filter
    """

    def __init__(self, f, h, Q, R, x0, P0, alpha=1e-3, beta=2, kappa=0):
        """
        Parameters
        ----------
        f : callable
            State dynamics: f(x, *args) -> x_dot (16-vector).
        h : callable
            Measurement function: h(x, *args) -> y.
        Q : callable
            Process noise covariance factory: Q(dt) -> ndarray (n, n).
        R : ndarray, shape (m, m)
            Measurement noise covariance.
        x0 : tuple[DualPose, DualVelocity]
            Initial state estimate on the dual quaternion manifold.
        P0 : ndarray, shape (n, n)
            Initial state covariance in the tangent space.
        alpha : float, optional
            UKF spread parameter (default 1e-3).
        beta : float, optional
            Prior knowledge parameter; 2 is optimal for Gaussian distributions.
        kappa : float, optional
            Secondary scaling parameter (default 0).
        """
        # Type / callable assertions
        assert callable(f), "f must be a callable: state [, *args] -> state derivative."
        assert callable(h), "h must be a callable: state [, *args] -> measurement."
        assert callable(Q), "Q must be a callable: Q(dt) -> process noise covariance."

        # R checks (measurement noise)
        assert isinstance(R, np.ndarray), "R must be a numpy array."
        assert R.ndim == 2 and R.shape[0] == R.shape[1], \
            "R must be a square covariance matrix."
        
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self.n = P0.shape[0]
        assert isinstance(x0[0], qn.DualPose)
        assert isinstance(x0[1], qn.DualVelocity)
        assert P0.shape == (self.n, self.n)
        self.P = P0
        self.x = x0
        
        self.m = None
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        self.n_sig = 2 * self.n + 1
        
        self.innov = None
        self.Pyy = None  # corrected name to match later usage

        # weights
        self.w_m = np.full((self.n_sig), 1.0 / (2 * (self.n + self.lambda_)))
        self.w_m[0] = self.lambda_ / (self.n + self.lambda_)
        self.w_c = self.w_m.copy()
        self.w_c[0] += 1 - alpha**2 + beta
    
    def sigma_points(self):
        """
        Compute 2n+1 sigma points in the tangent space about the current estimate.

        Returns
        -------
        sigma : ndarray, shape (2n+1, n)
            Sigma point offsets in the tangent space (mean is zero by construction).
        """
        # sigma points in tangent space (mean is 0 by construction)
        sqrt_P = sqrtm((self.n + self.lambda_) * self.P)
        assert not np.any(np.iscomplex(sqrt_P)), 'Sigma points have gone imaginary.'
        sigma = np.vstack([np.zeros((1, self.n)), sqrt_P, -sqrt_P])

        return sigma

    def predict(self, dt, *args):
        """
        Propagate sigma points through the dynamics and update the prior.

        Parameters
        ----------
        dt : float
            Integration time step in seconds.
        *args
            Additional arguments forwarded to the dynamics function f.
        """
        sigma = self.sigma_points()  # shape (n, n_sig)
        
        # propagate all sigma points through f
        sigma_prop_err = np.empty_like(sigma)
        sigma_prop_dq = np.empty((self.n_sig, 2), dtype=object)
        for i in range(self.n_sig):
            sigma_i = sigma[i]
            
            # # transform sigma point to manifold and perturb about estimate
            x_dq_i_old = qn.retract(self.x, sigma_i)
            x0_ode = np.concatenate((x_dq_i_old[0].dq, x_dq_i_old[1].dq))
            
            # numerically integrate
            sol = solve_ivp(lambda t, x: self.f(x, *args), [0, dt], x0_ode)
            assert sol.status == 0, 'Solver failed to converge.'
            
            # normalize + vectorize sigma points on manifold
            x = sol.y.T[-1]
            dq_tc = qn.DualQuaternion(dq=x[:8]).normalize()
            dw_tc_c = qn.DualQuaternion(dq=x[8:]).vectorize()
            sigma_prop_dq[i] = np.array([dq_tc, dw_tc_c])
            
            # transform DQ errors back to tangent space wrt DQ mean
            sigma_prop_err[i] = qn.lift(sigma_prop_dq[0], sigma_prop_dq[i])
        assert np.allclose(sigma_prop_err[0], np.zeros(12))
            
        # predicted covariance
        dx = sigma_prop_err
        P_hat = dx.T @ np.diag(self.w_c) @ dx + self.Q(dt)
        
        # update estimate in dq space
        self.x = sigma_prop_dq[0]
        self.P = P_hat

    def update(self, y, *args, **kwargs):
        """
        Correct the state estimate given a measurement.

        Parameters
        ----------
        y : ndarray, shape (m,)
            Measurement vector.
        *args, **kwargs
            Additional arguments forwarded to the measurement function h.

        Returns
        -------
        innov : ndarray, shape (m,)
            Innovation (measurement residual).
        Pyy : ndarray, shape (m, m)
            Innovation covariance.
        """
        self.m = y.size
        sigma = self.sigma_points()
        
        sigma_y = np.empty((self.n_sig, self.m))
        
        for i in range(self.n_sig):
            # calculate sigma points on manifold
            x_dq_i = qn.retract(self.x, sigma[i])
            
            # push sigma points (dq) through measurement function
            sigma_y[i] = self.h(x_dq_i, *args, **kwargs)
        
        # predicted measurement mean
        y_hat = self.w_m @ sigma_y
        
        # innovation covariance
        dy = sigma_y - y_hat
        Pyy = dy.T @ np.diag(self.w_c.ravel()) @ dy + self.R
        
        # cross covariance
        dx = sigma
        Pxy = dx.T @ np.diag(self.w_c) @ dy

        # Kalman gain
        K = Pxy @ np.linalg.inv(Pyy)
        
        # innovation
        innov = y - y_hat
        
        # correct nominal state (quaternion multiplicatively, other states additively)
        self.x = qn.retract(self.x, K @ innov)
        
        # update
        self.P = self.P - K @ Pyy @ K.T
        self.innov = innov
        self.Pyy = Pyy
        
        return innov, Pyy
    
    def likelihood(self):
        """Return the Gaussian likelihood of the most recent measurement update."""
        if self.innov is None or self.Pyy is None:
            raise ValueError("No measurement has been processed yet.")

        det_S = np.linalg.det(self.Pyy)
        if det_S <= 0:
            raise ValueError("Innovation covariance is not positive definite.")

        exponent = -0.5 * self.innov.T @ np.linalg.inv(self.Pyy) @ self.innov
        normalizer = np.sqrt((2 * np.pi) ** self.m * det_S)
        return np.squeeze(np.exp(exponent) / normalizer)