"""
DQ-UKF pose estimation simulation.

Simulates a 5-second trajectory in which a camera observes a target object
equipped with four square-arranged visual markers.  Synthetic pixel measurements
(with noise and dynamic occlusion) are fused via the DQ-UKF and compared
against an OpenCV solvePnP baseline.  Run this file directly to reproduce
the paper figures.
"""
import quaternion as qn
import numpy as np
from scipy.integrate import solve_ivp
import cv2
import matplotlib.pyplot as plt
import plotters as pf
from dqukf import DQUKF

def h(x, r_mt_t, K):
    """
    Camera measurement function.

    Projects 3D marker positions (in the target frame) into pixel coordinates
    using the relative pose contained in state x.

    Parameters
    ----------
    x : array of [DualPose, DualVelocity]
        Current state: relative pose dq_tc and velocity dw_tc_c.
    r_mt_t : ndarray, shape (n_markers, 3)
        Marker positions expressed in the target frame.
    K : ndarray, shape (3, 3)
        Camera intrinsic matrix.

    Returns
    -------
    ndarray, shape (2 * n_markers,)
        Flattened pixel coordinates [u1, v1, u2, v2, ...].
    """
    n_markers = r_mt_t.shape[0]

    dq_tc = x[0]
    assert isinstance(dq_tc, qn.DualPose)
    q_tc, r_tc_c = dq_tc.q_xy, dq_tc.r_xy_y
    r_mt_t = np.insert(r_mt_t, 0, 0, axis=1)
        
    # rotate markers from target frame to camera frame, then offset by translation
    rot = q_tc.left_mat() @ q_tc.conj().right_mat()
    r_mc_c = (np.repeat(np.expand_dims(r_tc_c.q, axis=1), n_markers, axis=1)
              + rot @ r_mt_t.T)
    r_mc_c = r_mc_c[1:, :].T
        
    # project points of markers relative to camera in camera frame
    pixels = cv2.projectPoints(r_mc_c, np.zeros(3), np.zeros(3), K, None)[0]   
    return pixels.flatten()

def f_ukf(x, dw_cj_c, dot_dw_tj_t, dot_dw_cj_c):
    """
    UKF process model: relative pose dynamics as a flat state derivative.

    Computes d/dt [dq_tc, dw_tc_c] using the camera-target relative dynamics,
    assuming the camera and target inertial motions are known (zero-order hold
    between filter updates).

    Parameters
    ----------
    x : ndarray, shape (16,)
        Concatenated dual quaternion state [dq_tc.dq, dw_tc_c.dq].
    dw_cj_c : DualVelocity
        Camera velocity w.r.t. inertial, expressed in the camera frame.
    dot_dw_tj_t : DualQuaternion
        Target angular acceleration w.r.t. inertial, in target frame.
    dot_dw_cj_c : DualQuaternion
        Camera angular acceleration w.r.t. inertial, in camera frame.

    Returns
    -------
    ndarray, shape (16,)
        Time derivative of the concatenated state.
    """
    assert isinstance(dw_cj_c, qn.DualVelocity)
    assert isinstance(dot_dw_tj_t, qn.DualQuaternion)
    assert isinstance(dot_dw_cj_c, qn.DualQuaternion)
    
    # unpack state and normalize/vectorize
    dq_tc, dw_tc_c = qn.array2dq(x, correct=False)
    
    # equations of motion
    dot_dq_tc = qn.kinematics(dq_tc, dw_tc_c)
    dot_dw_tc_c = qn.relative_dynamics(
        dq_tc, dw_tc_c, dw_cj_c, dot_dw_tj_t, dot_dw_cj_c)
    return qn.dq2array(dot_dq_tc, dot_dw_tc_c)

def f(x, df_x, M_x, M_x_inv=None):
    """
    Rigid body dynamics: compute the state derivative for a single body.

    Parameters
    ----------
    x : ndarray, shape (16,)
        Concatenated dual quaternion state [dq.dq, dw.dq].
    df_x : DualForce
        Applied wrench (force and torque) expressed in the body frame.
    M_x : ndarray, shape (8, 8)
        Mass/inertia matrix in dual quaternion form.
    M_x_inv : ndarray, shape (8, 8), optional
        Pre-computed inverse of M_x.  Computed internally if not provided.

    Returns
    -------
    ndarray, shape (16,)
        Time derivative of the concatenated state.
    """
    # unpack state
    dq_xy, dw_xy_x = qn.array2dq(x)
    
    # equations of motion
    dot_dq_xy = qn.kinematics(dq_xy, dw_xy_x=dw_xy_x)
    dot_dw_xy_x = qn.dynamics(dw_xy_x, df_x, M_x, M_x_inv=M_x_inv)
    return qn.dq2array(dot_dq_xy, dot_dw_xy_x)

def f_dt(dt, x, *args):
    """
    Discrete-time rigid body propagation via numerical integration.

    Integrates the continuous dynamics f() over interval [0, dt] using
    scipy.integrate.solve_ivp, then re-normalizes the pose and re-vectorizes
    the velocity.

    Parameters
    ----------
    dt : float
        Time step (s).
    x : array of [DualPose, DualVelocity]
        Current state.
    *args
        Additional arguments forwarded to f() (df_x, M_x, M_x_inv).

    Returns
    -------
    array of [DualPose, DualVelocity]
        Propagated state after dt seconds.
    """
    x0 = qn.state2array(x)
    sol = solve_ivp(lambda t, x: f(x, *args), [0, dt], x0)
    x_out = sol.y.T[-1]
    
    # unpack state and normalize/vectorize
    dq_xy, dw_xy_x = qn.array2dq(x_out, correct=True)   
    return qn.dq2state(dq_xy, dw_xy_x)

def square_points(length=1):
    """
    Return 4 marker positions at the corners of a square in the XY plane.

    Parameters
    ----------
    length : float
        Side length of the square (m).

    Returns
    -------
    ndarray, shape (4, 3)
        Marker 3D positions [x, y, 0] with z=0.
    """
    points = 1/2 * np.array([
        [length, length, 0],
        [-length, length, 0],
        [length, -length, 0],
        [-length, -length, 0]])
    return points

if __name__ == "__main__":
    np.random.seed(3)

    # simulation properties
    t_span = [0, 5]
    f_samp = 30 # Hz, sampling frequency
    dt = 1 / f_samp
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    n_pts = t_eval.size
    
    # markers/features
    r_mt_t = square_points(length=2)
    n_markers = r_mt_t.shape[0]

    # measurement properties
    K = np.array([[800, 0, 640],
                  [0, 800, 512],
                  [0, 0, 1]])
    R = 2**2 * np.eye(2 * n_markers) # pixel noise

    # process noise
    Q_a = 0**2
    Q_r = 0**2
    Q_w = 0.3**2
    Q_v = 0.3**2
    Q = lambda dt: dt * np.diag([Q_a, Q_a, Q_a, 
                                 Q_r, Q_r, Q_r,
                                 Q_w, Q_w, Q_w,
                                 Q_v, Q_v, Q_v])
    
    # target inertial state
    M_t = qn.build_M(1, np.diag([0.01, 0.02, 0.02]))
    M_t_inv = np.linalg.inv(M_t)
    q_tj = qn.qone()
    r_tj_t = qn.Quaternion(np.array([0, 0, 0, 10]))
    w_tj_t = qn.Quaternion(np.array([0, 0, 0.1, 0.1]))
    v_tj_t = qn.Quaternion(np.array([0, 0, 0, -1]))
    dq_tj = qn.DualPose(q_tj, r_xy_x = r_tj_t)
    dw_tj_t = qn.DualVelocity(v_xy_z=v_tj_t, w_xy_z=w_tj_t, r_xz_z=qn.qzero())
    df_t = qn.DualForce(dq=qn.dqzero().dq)
    x_tj_t = qn.dq2state(dq_tj, dw_tj_t)
    
    # camera inertial state
    M_c = qn.build_M(2, np.diag([0.03, 0.02, 0.01]))
    M_c_inv = np.linalg.inv(M_c)
    q_cj = qn.qone()
    r_cj_c = qn.Quaternion(np.array([0, -5, 0, 0]))
    w_cj_c = qn.Quaternion(np.array([0, 0, 0, 0]))
    v_cj_c = qn.Quaternion(np.array([0, 0, 0, 0]))
    dq_cj = qn.DualPose(q_cj, r_xy_x = r_cj_c)
    dw_cj_c = qn.DualVelocity(v_xy_z=v_cj_c, w_xy_z=w_cj_c, r_xz_z=qn.qzero())
    df_c = qn.DualForce(
        force=qn.Quaternion(np.array([0, 0.5, 0, 0])),
        torque=qn.Quaternion([0, 0, 0.0005, 0]))
    x_cj_c = qn.dq2state(dq_cj, dw_cj_c)

    # define true initial state
    dq_tc = dq_cj.conj() * dq_tj
    dw_tc_c = dq_tc * dw_tj_t * dq_tc.conj() - dw_cj_c
    x_tc_c_truth = qn.dq2state(dq_tc, dw_tc_c)

    # initial state estimate
    P_a = 0.2**2 # rad
    P_r = 0.2**2 # m
    P_w = 0.1**2 # rad/s
    P_v = 0.5**2 # m/s
    P0 = np.diag([P_a, P_a, P_a,
                  P_r, P_r, P_r,
                  P_w, P_w, P_w,
                  P_v, P_v, P_v])
    delta0_perturb = np.random.multivariate_normal(np.zeros(12), P0)
    x0_est = qn.retract(x_tc_c_truth, delta0_perturb)
    
    test = qn.lift(x_tc_c_truth, x_tc_c_truth)
    assert np.allclose(test, np.zeros(12))
    delta_test = qn.lift(x_tc_c_truth, x0_est)
    assert np.allclose(delta_test, delta0_perturb)
    assert np.allclose(x0_est[0].q_xy.log(), delta0_perturb[:3])
    assert np.allclose(
        x0_est[0].r_xy_y.qvec - x_tc_c_truth[0].r_xy_y.qvec,
        delta0_perturb[3:6])
    r_xy_y_est = x0_est[0].r_xy_y
    assert np.allclose(
        x0_est[1].w_xy_z.qvec - x_tc_c_truth[1].w_xy_z.qvec,
        delta0_perturb[6:9])
    v_est = x0_est[1].v_xy_z(r_xy_y_est).qvec
    v_truth = x_tc_c_truth[1].v_xy_z(x_tc_c_truth[0].r_xy_y).qvec
    assert np.allclose(v_est - v_truth, delta0_perturb[9:])
    
    # preallocate
    hist_x_cj = np.empty((n_pts, 2), dtype=object)
    hist_x_tj = np.empty((n_pts, 2), dtype=object)
    hist_x_tc_truth = np.empty((n_pts, 2), dtype=object)
    hist_x_tc_est = np.empty((n_pts, 2), dtype=object)
    hist_x_tc_cv = np.empty((n_pts), dtype=object)
    hist_dot_dw_tc_c = np.empty((n_pts), dtype=object)
    
    hist_log_cj = np.empty((n_pts, 12))
    hist_log_tj = np.empty((n_pts, 12))
    hist_log_tc_truth = np.empty((n_pts, 12))
    hist_log_tc_est = np.empty((n_pts, 12))
    hist_log_tc_est_err = np.empty((n_pts, 12))
    hist_log_tc_cv = np.empty((n_pts, 6))
    hist_log_tc_cv_err = np.empty((n_pts, 6))
    hist_log_n_markers = np.empty((n_pts))
    
    hist_P_est = np.empty((n_pts, 12, 12))
    hist_y = np.empty((n_pts, n_markers, 2))
    
    for i in range(n_pts):
        dq_tj, dw_tj_t = qn.state2dq(x_tj_t)
        dq_cj, dw_cj_c = qn.state2dq(x_cj_c)
        
        # truth state
        dq_tc = dq_cj.conj() * dq_tj
        dw_tc_c = dq_tc * dw_tj_t * dq_tc.conj() - dw_cj_c
        x_tc_c_truth = qn.dq2state(dq_tc, dw_tc_c)
        
        # get noisy measurements
        y_clean = h(x_tc_c_truth, r_mt_t, K)
        
        # add noise to measurement before passing to pf
        y = np.random.multivariate_normal(y_clean, R)
        
        # occlusion
        if (t_eval[i] >= 1 and t_eval[i] <= 2) or (t_eval[i] >= 3 and t_eval[i] <= 4):
            n_markers = 3
        elif t_eval[i] >= 2 and t_eval[i] <= 3:
            n_markers = 2
        else:
            n_markers = 4
        y_update = y[:2 * n_markers]
        r_mt_t_update = r_mt_t[:n_markers]
        hist_log_n_markers[i] = y_update.size / 2
            
        # solve pnp with opencv
        if n_markers >= 3:
            success, rvecs, tvecs = cv2.solvePnP(
                r_mt_t_update,
                np.reshape(y_update, (-1, 2)),
                K, None, flags=cv2.SOLVEPNP_SQPNP)
            assert np.ndim(rvecs) == 2, 'not built for multiple solutions yet'
            
            # construct opencv estimate
            t_vec = np.concatenate((np.zeros(1), tvecs.flatten()))
            dq_tc_cv = qn.DualPose(
                q_xy=qn.UnitQuaternion(rvecs.flatten()),
                r_xy_y=qn.Quaternion(t_vec))
        else:
            # PnP requires at least 3 markers; use identity as placeholder.
            dq_tc_cv = qn.dqone()
        
        # update estimate or initialize filter
        if i == 0:
            # initialize with pnp estimate
            delta0_perturb = np.random.multivariate_normal(np.zeros(12), P0)
            # zero out pose errors: initialize from PnP estimate directly
            delta0_perturb[:6] = np.zeros(6)
            x0_init = np.array([dq_tc_cv, x_tc_c_truth[1]])
            x0_est = qn.retract(x0_init, delta0_perturb)
            
            ukf = DQUKF(f_ukf, h, Q=Q, R=R, x0=x0_est, P0=P0)
        else:
            ukf.R = R[:2 * n_markers, :2 * n_markers]
            ukf.update(y_update, r_mt_t_update, K)
        
        # save truth, estimate, and cv states
        hist_x_cj[i] = x_cj_c
        hist_x_tj[i] = x_tj_t
        hist_x_tc_truth[i] = x_tc_c_truth
        hist_x_tc_est[i] = ukf.x
        hist_P_est[i, :, :] = ukf.P
        hist_x_tc_cv[i] = dq_tc_cv
        
        # save log of states
        hist_log_cj[i] = qn.log(x_cj_c, x_frame=True)
        hist_log_tj[i] = qn.log(x_tj_t, x_frame=True)
        hist_log_tc_truth[i] = qn.log(x_tc_c_truth, x_frame=False)
        hist_log_tc_est[i] = qn.log(ukf.x, x_frame=False)
        hist_log_tc_cv[i] = dq_tc_cv.log(x_frame=False)
        hist_log_tc_est_err[i] = qn.lift(x_tc_c_truth, ukf.x)
        hist_log_tc_cv_err[i] = qn.lift(x_tc_c_truth[0], dq_tc_cv)
        
        # save measurements
        hist_y[i] = np.reshape(y, (-1, 2))
        
        # prop estimate and assume ZOH on camera and target wrt inertial states
        dot_dw_tj_t = qn.dynamics(dw_tj_t, df_t, M_t, M_t_inv) 
        dot_dw_cj_c = qn.dynamics(dw_cj_c, df_c, M_c, M_c_inv) 
        dot_dw_tc_c = qn.relative_dynamics(
            dq_tc, dw_tc_c, dw_cj_c, dot_dw_tj_t, dot_dw_cj_c)
        ukf.predict(dt, dw_cj_c, dot_dw_tj_t, dot_dw_cj_c)
        hist_dot_dw_tc_c[i] = dot_dw_tc_c
        
        # propagate truth inertial states and estimate
        x_tj_t = f_dt(dt, x_tj_t, df_t, M_t, M_t_inv)
        x_cj_c = f_dt(dt, x_cj_c, df_c, M_c, M_c_inv)
        
    
    # calculate errors
    est_error_mag = np.concatenate((
        np.linalg.norm(hist_log_tc_est_err[:, :3], axis=1, keepdims=True),
        np.linalg.norm(hist_log_tc_est_err[:, 3:6], axis=1, keepdims=True),
    ), axis=1)
    cv_error_mag = np.concatenate((
        np.linalg.norm(hist_log_tc_cv_err[:, :3], axis=1, keepdims=True),
        np.linalg.norm(hist_log_tc_cv_err[:, 3:6], axis=1, keepdims=True),
    ), axis=1)
    # plot truth
    xlabel = 'Time (s)'
    state_labels = [
        r'$\overline{\phi}_x$', r'$\overline{\phi}_y$', r'$\overline{\phi}_z$',
        r'$\overline{r}_x$', r'$\overline{r}_y$', r'$\overline{r}_z$',
        r'$\overline{\omega}_x$', r'$\overline{\omega}_y$', r'$\overline{\omega}_z$',
        r'$\overline{v}_x$', r'$\overline{v}_y$', r'$\overline{v}_z$']
    error_labels = [r'$||\overline{\phi}||$ (rad)', r'$||\overline{r}||$ (m)']
    pf.plot_states(
        t_eval, hist_log_cj, title='Camera wrt Inertial in Camera',
        xlabel=xlabel, ylabel=state_labels)
    pf.plot_states(
        t_eval, hist_log_tc_truth,
        xlabel=xlabel, ylabel=state_labels, figsize=(6.4, 9))
    pf.plot_estimates(
        t_eval, hist_log_tc_est_err, hist_P_est,
        leglabel='Residual', xlabel=xlabel, ylabel=state_labels,
        figsize=(6.4, 9))
    pf.plot_state_comparison(
        t_eval, est_error_mag, cv_error_mag,
        marker_hist=hist_log_n_markers, ylabel=error_labels,
        xlabel=xlabel, labels=('DQ-UKF', 'OpenCV'))
    pf.plot_frames(
        hist_log_cj[:, :3], hist_log_cj[:, 3:6],
        hist_log_tj[:, :3], hist_log_tj[:, 3:6],
        draw_every=f_samp, figsize=(8, 8))
    plt.show()