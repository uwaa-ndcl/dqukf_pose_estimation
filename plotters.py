"""
Visualization utilities for DQ-UKF pose estimation results.

Provides functions for plotting state trajectories, estimation residuals,
confidence bounds, marker pixel streaks, and 3D coordinate frames.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Common style parameters
_LINEWIDTH = 1.5
_TITLE_FONTSIZE = 12
_LABEL_FONTSIZE = 12
_LEGEND_FONTSIZE = 12
_SAVEFIG_KW = dict(bbox_inches='tight', pad_inches=0.02)


def plot_states(time, states, title=None, ylabel=None, xlabel='Time',
                figsize=(6.4, 4.8), savefig=False, fname='states', ftype='.pdf'):
    """
    Plot state trajectories over time using compact constrained layout (no legend).

    Parameters
    ----------
    time : array-like, shape (T,)
        Time vector.
    states : array-like, shape (T, n)
        State trajectories.
    title : str, optional
        Figure title.
    ylabel : list of str, optional
        Per-subplot y-axis labels. Defaults to ``"State i"``.
    xlabel : str, optional
        Shared x-axis label (default ``'Time'``).
    figsize : tuple, optional
        Figure size in inches (default ``(6.4, 4.8)``).
    savefig : bool, optional
        Save figure to file if True (default False).
    fname : str, optional
        Output filename without extension (default ``'states'``).
    ftype : str, optional
        File extension including dot (default ``'.pdf'``).
    """
    states = np.asarray(states)
    T, n = states.shape

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, layout='constrained')
    if n == 1:
        axes = [axes]

    for i in range(n):
        axes[i].plot(time, states[:, i], linewidth=_LINEWIDTH)
        lbl = ylabel[i] if ylabel is not None else f"State {i}"
        axes[i].set_ylabel(lbl, fontsize=_LABEL_FONTSIZE)
        axes[i].set_xlim([np.min(time), np.max(time)])
        axes[i].tick_params(axis='both', which='major', labelsize=_LABEL_FONTSIZE)

    axes[-1].set_xlabel(xlabel, fontsize=_LABEL_FONTSIZE)
    if title is not None:
        fig.suptitle(title, fontsize=_TITLE_FONTSIZE)

    if savefig:
        fig.savefig(fname + ftype, **_SAVEFIG_KW)


def plot_estimates(time, residuals, covariances, title=None, ylabel=None,
                   xlabel='Time', leglabel='Estimate',
                   figsize=(6.4, 4.8), savefig=False,
                   fname='residuals', ftype='.pdf'):
    """
    Plot residuals with ±3σ confidence bounds using compact constrained layout.

    Parameters
    ----------
    time : array-like, shape (T,)
        Time vector.
    residuals : array-like, shape (T, n)
        Residual trajectories (estimate minus truth).
    covariances : array-like, shape (T, n, n)
        Covariance history used to compute ±3σ bounds.
    title : str, optional
        Figure title.
    ylabel : list of str, optional
        Per-subplot y-axis labels. Defaults to ``"State i"``.
    xlabel : str, optional
        Shared x-axis label (default ``'Time'``).
    leglabel : str, optional
        Legend label for the residual line (default ``'Estimate'``).
    figsize : tuple, optional
        Figure size in inches (default ``(6.4, 4.8)``).
    savefig : bool, optional
        Save figure to file if True (default False).
    fname : str, optional
        Output filename without extension (default ``'residuals'``).
    ftype : str, optional
        File extension including dot (default ``'.pdf'``).
    """
    residuals = np.asarray(residuals)
    covariances = np.asarray(covariances)
    
    T, n = residuals.shape
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, layout='constrained')
    if n == 1:
        axes = [axes]

    for i in range(n):
        mean = residuals[:, i]
        std = np.sqrt(covariances[:, i, i])

        line, = axes[i].plot(time, mean, label=leglabel, linewidth=_LINEWIDTH)
        axes[i].fill_between(
            time, -3 * std, 3 * std,
            color=line.get_color(), alpha=0.2, label=r'$3\sigma$')
        lbl = ylabel[i] if ylabel is not None else f"State {i}"
        axes[i].set_ylabel(lbl, fontsize=_LABEL_FONTSIZE)
        axes[i].set_xlim([np.min(time), np.max(time)])
        if i == 0:
            axes[i].legend(frameon=True, loc='upper right', fontsize=_LEGEND_FONTSIZE)
        axes[i].tick_params(axis='both', which='major', labelsize=_LABEL_FONTSIZE)

    axes[-1].set_xlabel(xlabel, fontsize=_LABEL_FONTSIZE)
    if title is not None:
        fig.suptitle(title, fontsize=_TITLE_FONTSIZE)

    if savefig:
        fig.savefig(fname + ftype, **_SAVEFIG_KW)


def plot_likelihood(time, likelihood, title='Likelihood', ylabel=None, xlabel='Time',
                    figsize=(6.4, 4.8), savefig=False, fname='likelihood', ftype='.pdf'):
    """
    Plot filter likelihoods over time using compact constrained layout (no legend).

    Parameters
    ----------
    time : array-like, shape (T,)
        Time vector.
    likelihood : array-like, shape (T,) or (k, T)
        Likelihood values; a 1-D array is treated as a single filter.
    title : str, optional
        Figure title (default ``'Likelihood'``).
    ylabel : str, optional
        Y-axis label. Defaults to ``'Likelihood'``.
    xlabel : str, optional
        X-axis label (default ``'Time'``).
    figsize : tuple, optional
        Figure size in inches (default ``(6.4, 4.8)``).
    savefig : bool, optional
        Save figure to file if True (default False).
    fname : str, optional
        Output filename without extension (default ``'likelihood'``).
    ftype : str, optional
        File extension including dot (default ``'.pdf'``).
    """
    likelihood = np.atleast_2d(likelihood)
    num_filters = likelihood.shape[0]

    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    for i in range(num_filters):
        ax.plot(time, likelihood[i], linewidth=_LINEWIDTH)

    ax.set_xlabel(xlabel, fontsize=_LABEL_FONTSIZE)
    lbl = ylabel if ylabel is not None else 'Likelihood'
    ax.set_ylabel(lbl, fontsize=_LABEL_FONTSIZE)
    fig.suptitle(title, fontsize=_TITLE_FONTSIZE)

    if savefig:
        fig.savefig(fname + ftype, **_SAVEFIG_KW)
        

def plot_state_comparison(time, states1, states2, marker_hist=None,
                          title=None, ylabel=None, xlabel='Time',
                          labels=('Method 1', 'Method 2'), colors=('C0', 'C1'),
                          figsize=(6.4, 4.8), savefig=False,
                          fname='state_comparison', ftype='.pdf'):
    """
    Compare two sets of state trajectories over time.

    Parameters
    ----------
    time : array-like, shape (T,)
        Time vector.
    states1 : array-like, shape (T, n)
        First set of state trajectories (e.g. filter estimate).
    states2 : array-like, shape (T, n)
        Second set of state trajectories (e.g. baseline). Timesteps where
        ``marker_hist <= 2`` are masked to NaN when ``marker_hist`` is provided.
    marker_hist : array-like, shape (T,), optional
        Number of visible markers at each timestep. If provided, an extra subplot
        is appended and ``states2`` is masked where the count is ≤ 2.
    title : str, optional
        Figure title.
    ylabel : list of str, optional
        Per-subplot y-axis labels. Defaults to ``"State i"``.
    xlabel : str, optional
        Shared x-axis label (default ``'Time'``).
    labels : tuple of str, optional
        Legend labels for states1 and states2 (default ``('Method 1', 'Method 2')``).
    colors : tuple of str, optional
        Line colors for states1 and states2 (default ``('C0', 'C1')``).
    figsize : tuple, optional
        Figure size in inches (default ``(6.4, 4.8)``).
    savefig : bool, optional
        Save figure to file if True (default False).
    fname : str, optional
        Output filename without extension (default ``'state_comparison'``).
    ftype : str, optional
        File extension including dot (default ``'.pdf'``).
    """
    states1 = np.asarray(states1)
    states2 = np.asarray(states2)
    T, n = states1.shape
    assert states2.shape == (T, n), "states1 and states2 must have the same shape"

    # Mask states2 if marker_hist values are <= 2
    if marker_hist is not None:
        marker_hist = np.asarray(marker_hist)
        mask = marker_hist <= 2
        states2_plot = states2.copy()
        states2_plot[mask, :] = np.nan
    else:
        states2_plot = states2

    n_rows = n if marker_hist is None else n + 1
    fig, axes = plt.subplots(
        n_rows, 1, figsize=figsize, sharex=True, layout='constrained')
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        ax.plot(time, states1[:, i], color=colors[0], label=labels[0], lw=_LINEWIDTH)
        ax.plot(time, states2_plot[:, i], color=colors[1], label=labels[1],
                lw=_LINEWIDTH)
        lbl = ylabel[i] if ylabel is not None else f"State {i}"
        ax.set_ylabel(lbl, fontsize=_LABEL_FONTSIZE)
        ax.set_xlim([np.min(time), np.max(time)])
        if i == 0:
            ax.legend(frameon=True, loc='upper right', fontsize=_LEGEND_FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=_LABEL_FONTSIZE)
            
    if marker_hist is not None:
        ax = axes[-1]
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(time, marker_hist, lw=_LINEWIDTH)
        ax.set_ylabel('# of Markers', fontsize=_LABEL_FONTSIZE)
        ax.set_xlim([np.min(time), np.max(time)])
        ax.tick_params(axis='both', which='major', labelsize=_LABEL_FONTSIZE)
    
    axes[-1].set_xlabel(xlabel, fontsize=_LABEL_FONTSIZE)
    if title is not None:
        fig.suptitle(title, fontsize=_TITLE_FONTSIZE)

    if savefig:
        fig.savefig(fname + ftype, **_SAVEFIG_KW)

def plot_marker_streaks(pixels, K=None, figsize=(6, 6), title="Marker Pixel Streaks",
                        color='tab:blue'):
    """
    Plot pixel streaks over time for each marker, using a single color for all.
    Start points are green circles, end points are red crosses, and if a camera
    calibration matrix K is provided, the optical center is plotted as a black star.

    Parameters
    ----------
    pixels : np.ndarray
        Array of shape (n_time_steps, n_markers, 2) with [x, y] pixel coordinates.
    K : np.ndarray, optional
        Camera calibration matrix (3x3). If provided, plots the optical center (cx, cy).
    figsize : tuple
        Figure size in inches.
    title : str
        Plot title.
    color : str
        Color used for all marker trajectories.
    """
    n_time, n_markers, _ = pixels.shape

    plt.figure(figsize=figsize, layout='constrained')
    ax = plt.gca()

    # Plot marker trajectories
    for i in range(n_markers):
        traj = pixels[:, i, :]
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, alpha=0.7)
        ax.scatter(traj[0, 0], traj[0, 1], color='green', marker='o', s=30)  # start
        ax.scatter(traj[-1, 0], traj[-1, 1], color='red', marker='x', s=40)  # end

    # Plot the optical center if camera matrix provided
    if K is not None and K.shape == (3, 3):
        cx, cy = K[0, 2], K[1, 2]
        ax.scatter(cx, cy, color='black', marker='*', s=80)  # optical center

    # Match image coordinate convention and scaling
    ax.set_aspect('equal', adjustable='box') # equal scaling

    # Labels, grid, legend
    ax.set_xlabel("Pixel X", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Pixel Y", fontsize=_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=_TITLE_FONTSIZE)
    ax.grid(True)

    # Legend handles (dummy plots for clean labeling)
    legend_handles = [
        plt.Line2D([], [], color=color, lw=2, label='Marker Streak'),
        plt.Line2D([], [], color='green', marker='o',
                   linestyle='None', markersize=6, label='Start'),
        plt.Line2D([], [], color='red', marker='x',
                   linestyle='None', markersize=6, label='End'),
    ]
    if K is not None:
        legend_handles.append(
            plt.Line2D([], [], color='black', marker='*',
                       linestyle='None', markersize=8, label='Optical Center')
        )

    ax.legend(handles=legend_handles, loc='best', fontsize=_LEGEND_FONTSIZE)


def plot_frames(
    r_ci, p_c_body, r_ti, p_t_body, 
    draw_every=5, arrow_fraction=0.1,
    figsize=(7, 7), savefig=False, fname='frames', ftype='.pdf'):
    """
    Plot camera and target frames in 3D over time using body-frame positions and
    Rodrigues parameters (rotation vectors) for rotation, both defined w.r.t. the inertial frame.

    Parameters
    ----------
    p_c_body : np.ndarray, shape (T,3)
        Camera positions w.r.t. inertial frame, expressed in camera body frame.
    r_ci : np.ndarray, shape (T,3)
        Camera orientations as Rodrigues parameters (rotation vectors) w.r.t. inertial frame.
    p_t_body : np.ndarray, shape (T,3)
        Target positions w.r.t. inertial frame, expressed in target body frame.
    r_ti : np.ndarray, shape (T,3)
        Target orientations as Rodrigues parameters (rotation vectors) w.r.t. inertial frame.
    draw_every : int
        Draw frames every `draw_every` timesteps.
    arrow_fraction : float
        Fraction of axis range used for arrow length.
    title : str
        Plot title.
    """

    T = p_c_body.shape[0]

    # Convert Rodrigues parameters (rotation vectors) to rotation matrices
    R_ci = R.from_rotvec(r_ci).as_matrix()  # inertial <- camera
    R_ti = R.from_rotvec(r_ti).as_matrix()  # inertial <- target

    # Convert body-frame positions to inertial coordinates
    p_ci = np.einsum('tij,tj->ti', R_ci, p_c_body)
    p_ti = np.einsum('tij,tj->ti', R_ti, p_t_body)

    # --- Setup figure ---
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectories
    ax.plot(p_ci[:, 0], p_ci[:, 1], p_ci[:, 2], "b-", label="Camera")
    ax.plot(p_ti[:, 0], p_ti[:, 1], p_ti[:, 2], "r-", label="Target")

    # Axis limits
    all_points = np.vstack([p_ci, p_ti])
    min_vals, max_vals = all_points.min(axis=0), all_points.max(axis=0)
    ax.set_xlim(min_vals[0]-0.2, max_vals[0]+0.2)
    ax.set_ylim(min_vals[1]-0.2, max_vals[1]+0.2)
    ax.set_zlim(min_vals[2]-0.2, max_vals[2]+0.2)

    # Arrow length
    axis_range = max(max_vals - min_vals)
    arrow_length = axis_range * arrow_fraction

    # Draw frames
    for t in range(0, T, draw_every):
        draw_frame_lines(ax, p_ci[t], R_ci[t], arrow_length, is_camera=True)
        draw_frame_lines(ax, p_ti[t], R_ti[t], arrow_length, is_camera=False)

    # Labels
    ax.set_xlabel("X (m)", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Y (m)", fontsize=_LABEL_FONTSIZE)
    ax.set_zlabel("Z (m)", fontsize=_LABEL_FONTSIZE)
    ax.legend(fontsize=_LEGEND_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=_LABEL_FONTSIZE)
    set_axes_equal(ax)
    
    if savefig:
        fig.savefig(fname + ftype, bbox_inches='tight', pad_inches=0.22)


def draw_frame_lines(ax, origin, R_mat, length, is_camera=False):
    """
    Draw a 3D coordinate frame at ``origin`` with rotation matrix ``R_mat``.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axes to draw on.
    origin : array-like, shape (3,)
        Frame origin in inertial coordinates.
    R_mat : ndarray, shape (3, 3)
        Rotation matrix whose columns are the frame axes in inertial coordinates.
    length : float
        Arrow length in the same units as the axes.
    is_camera : bool, optional
        If True, additionally draw a filled frustum representing a camera
        (default False).
    """

    # -------- Axis Colors --------
    if is_camera:
        # lighter → darker blues
        colors = ["#88c5ff", "#3399ff", "#004c99"]
    else:
        # lighter → darker reds
        colors = ["#ff9999", "#ff4d4d", "#990000"]

    axes = np.eye(3)

    # -------- Draw +X, +Y, +Z axes --------
    for i in range(3):
        end = origin + length * R_mat[:, i]
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=colors[i],
            linewidth=2,
        )

    # -------- Marker at +X axis --------
    pos_x_end = origin + length * R_mat[:, 0]
    ax.scatter(pos_x_end[0], pos_x_end[1], pos_x_end[2],
               color=colors[0], s=40)

    # -------- If target frame, stop here --------
    if not is_camera:
        return

    # ======================================================
    #                   CAMERA FRUSTUM
    # ======================================================

    # Frustum geometry
    near = 0.4 * length
    far = 1.0 * length
    half_size_near = 0.15 * length
    half_size_far  = 0.35 * length

    # Local coordinates in camera frame
    near_local = np.array([
        [-half_size_near, -half_size_near, near],
        [ half_size_near, -half_size_near, near],
        [ half_size_near,  half_size_near, near],
        [-half_size_near,  half_size_near, near],
    ])

    far_local = np.array([
        [-half_size_far, -half_size_far, far],
        [ half_size_far, -half_size_far, far],
        [ half_size_far,  half_size_far, far],
        [-half_size_far,  half_size_far, far],
    ])

    # Transform to inertial frame
    near_global = origin + (R_mat @ near_local.T).T
    far_global  = origin + (R_mat @ far_local.T).T

    # ------------ Filled Faces ------------
    faces = [
        near_global,                          # near plane
        far_global,                           # far plane
        np.array([near_global[0], near_global[1], far_global[1], far_global[0]]),
        np.array([near_global[1], near_global[2], far_global[2], far_global[1]]),
        np.array([near_global[2], near_global[3], far_global[3], far_global[2]]),
        np.array([near_global[3], near_global[0], far_global[0], far_global[3]]),
    ]

    frustum_faces = Poly3DCollection(
        faces,
        facecolors="lightskyblue",
        edgecolor="k",
        linewidths=0.8,
        alpha=0.25                 # transparent fill
    )
    ax.add_collection3d(frustum_faces)


def set_axes_equal(ax):
    """Make 3D plots have equal scale on all axes."""
    limits = np.array([
        ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    ])
    centers = limits.mean(axis=1)
    max_range = (limits[:, 1] - limits[:, 0]).max() / 2
    for center, axis in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        axis(center - max_range, center + max_range)