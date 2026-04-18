"""
Quaternion and dual quaternion algebra for rigid body pose representation.

Provides classes for unit quaternions (rotations), dual quaternions (poses),
dual velocity, and dual force, together with kinematics, dynamics, and
manifold operations (log, exp, retract, lift) used by the DQ-UKF.
"""
import numpy as np


class Quaternion:
    __array_priority__ = 10000

    def __init__(self, q):
        q = np.array(q)
        assert q.shape == (4,), "Quaternion must be a 4-vector"
        self._q = q  # protected internal array

    # -------------------------------------------------------------------------
    # Core properties
    # -------------------------------------------------------------------------
    @property
    def q(self):
        """Return the underlying quaternion array (read-only)."""
        return self._q

    @property
    def q0(self):
        return self._q[0]

    @property
    def qvec(self):
        return self._q[1:]

    # -------------------------------------------------------------------------
    # Representation and indexing
    # -------------------------------------------------------------------------
    def __getitem__(self, key):
        return self._q[key]

    def __repr__(self):
        return np.array_str(self._q)

    # -------------------------------------------------------------------------
    # Arithmetic operations
    # -------------------------------------------------------------------------
    def __add__(self, other):
        if isinstance(other, Quaternion):
            out = self._q + other._q
        elif isinstance(other, np.ndarray):
            out = self._q + other
        else:
            raise TypeError("Unsupported addition type")
        return Quaternion(out)

    def __sub__(self, other):
        if isinstance(other, Quaternion):
            out = self._q - other._q
        elif isinstance(other, np.ndarray):
            out = self._q - other
        else:
            raise TypeError("Unsupported subtraction type")
        return Quaternion(out)

    def __mul__(self, other):
        # Quaternion * Quaternion
        if isinstance(other, Quaternion):
            w1, v1 = self.q0, self.qvec
            w2, v2 = other.q0, other.qvec
            w = w1 * w2 - np.dot(v1, v2)
            v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
            return Quaternion(np.concatenate([np.array([w]), v]))
        # Scalar * Quaternion
        elif np.isscalar(other):
            return Quaternion(other * self._q)
        else:
            raise TypeError("Unsupported multiplication type")

    def __rmul__(self, other):
        # Defer to __mul__, but with operands swapped
        if np.isscalar(other):
            # Scalar multiplication is commutative
            return self.__mul__(other)
        elif isinstance(other, Quaternion):
            # Reverse the order of multiplication
            return other.__mul__(self)
        else:
            raise TypeError("Unsupported multiplication type")


    def __truediv__(self, other):
        if np.isscalar(other):
            return Quaternion(self._q / other)
        raise TypeError("Division only supported by scalars")

    def __neg__(self):
        return Quaternion(-self._q)

    def __eq__(self, other):
        return np.allclose(self._q, other._q)
    
    # -------------------------------------------------------------------------
    # Matrix (star) multiplication
    # -------------------------------------------------------------------------
    def __rmatmul__(self, matrix):
        """4x4 matrix @ Quaternion."""
        assert isinstance(matrix, np.ndarray) and matrix.shape == (4, 4), \
            "Left operand must be a numpy ndarray with shape (4, 4)."
        out = np.squeeze(matrix @ self._q[:, None])
        return Quaternion(out)


    # -------------------------------------------------------------------------
    # Quaternion operations
    # -------------------------------------------------------------------------
    def conj(self):
        return Quaternion(np.concatenate([np.array([self.q0]), -self.qvec]))

    def dot(self, other):
        assert isinstance(other, Quaternion)
        dot_val = self.q0 * other.q0 + np.dot(self.qvec, other.qvec)
        return Quaternion(np.array([dot_val, 0, 0, 0]))
    
    def norm_sq(self):
        return self.dot(self)

    def norm(self):
        return np.sqrt(self.norm_sq().q0)
    
    def normalize(self):
        q = UnitQuaternion(self._q / self.norm())
        assert q.isunit()
        return q
    
    def vectorize(self):
        return Quaternion(np.concatenate((np.zeros(1), self.qvec)))

    def cross(self, other):
        assert isinstance(other, Quaternion)
        v = (self.q0 * other.qvec + other.q0 * self.qvec
             + np.cross(self.qvec, other.qvec))
        return Quaternion(np.concatenate([np.array([0.0]), v]))

    # -------------------------------------------------------------------------
    # Matrix representations
    # -------------------------------------------------------------------------
    @staticmethod
    def _skew(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
    
    def left_mat(self):
        w, v = self.q0, self.qvec
        return np.block([
            [np.array([[w]]), -v[None, :]],
            [v[:, None], w * np.eye(3) + self._skew(v)],
        ])

    def right_mat(self):
        w, v = self.q0, self.qvec
        return np.block([
            [np.array([[w]]), -v[None, :]],
            [v[:, None], w * np.eye(3) - self._skew(v)],
        ])
        
    def cross_mat(self):
        q0, qv = self.q0, self.qvec
        return np.block([[np.zeros((1,1)), np.zeros((1, 3))],
                         [np.expand_dims(qv, axis=1), q0 * np.eye(3) + self._skew(qv)]])

    # -------------------------------------------------------------------------
    # Classification helpers
    # -------------------------------------------------------------------------
    def isunit(self):
        return np.allclose(self.norm_sq().q, np.array([1, 0, 0, 0]))

    def isvec(self):
        return np.allclose(self.q0, 0)

class UnitQuaternion(Quaternion):
    """Quaternion constrained to norm 1 (rotation)."""

    def __init__(self, q, method=None):
        """
        Create a UnitQuaternion.

        Parameters
        ----------
        q : array-like, Quaternion, or UnitQuaternion
            - If shape (4,), interpreted as [w, x, y, z].
            - If shape (3,), interpreted as a rotation vector.
            - If Quaternion, converted to unit form.
        """
        # Case 1: already a Quaternion
        if isinstance(q, Quaternion):
            q_arr = q.q

        # Case 2: array-like
        else:
            q = np.asarray(q, dtype=float)
            if q.shape == (3,):  # rotation vector
                # exponential map from rotation vector -> quaternion
                theta = np.linalg.norm(q)
                if theta < 1e-12:
                    q_arr = np.array([1.0, 0.0, 0.0, 0.0])
                else:
                    axis = q / theta
                    half_theta = theta / 2.0
                    q_arr = np.concatenate([
                        [np.cos(half_theta)],
                        axis * np.sin(half_theta)
                    ])
            elif q.shape == (4,):  # Quaternion vector
                q_arr = q
            else:
                raise ValueError(
                    "Input must be a 3-vector (rotation vector) "
                    "or 4-vector (quaternion)."
                )

        super().__init__(q_arr)
        
        assert self.isunit(), 'Input is not a unit quaternion.'

    def __mul__(self, other):
        result = super().__mul__(other)
        # Type-preserving multiplication
        if isinstance(other, UnitQuaternion):
            return UnitQuaternion(result.q)
        else:
            return result
        
    def __rmul__(self, other):
        result = super().__rmul__(other)
        # Type-preserving multiplication
        if isinstance(other, UnitQuaternion):
            return UnitQuaternion(result.q)
        else:
            return result
        
    def conj(self):
        return UnitQuaternion(np.concatenate([np.array([self.q0]), -self.qvec]))

    def log(self):
        # quaternion -> rotation vector
        v_norm = np.linalg.norm(self.qvec)
        if np.allclose(v_norm, 0):
            return np.zeros(3)
        return 2 * np.atan2(v_norm, self.q0) * self.qvec / v_norm

class DualQuaternion:
    __array_priority__ = 10000
    
    def __init__(self, q_real=None, q_dual=None, dq=None):
        if dq is not None:
            dq = np.array(dq)
            assert dq.shape == (8,), "Dual quaternion must be a 4-vector"
            self._dq = dq
        else:
            assert isinstance(q_real, Quaternion), 'q_real must be a Quaternion'
            assert isinstance(q_dual, Quaternion), 'q_dual must be a Quaternion'
            self._dq = np.concatenate((q_real.q, q_dual.q))
    
    # -------------------------------------------------------------------------
    # Core properties
    # -------------------------------------------------------------------------
    @property
    def dq(self):
        """Return the underlying quaternion array (read-only)."""
        return self._dq
    
    @property
    def dq0(self):
        return np.array([self._dq[0], self._dq[4]])
    
    @property
    def dqvec(self):
        return np.concatenate((self._dq[1:4], self._dq[5:]))

    @property
    def real(self):
        return Quaternion(self._dq[:4])

    @property
    def dual(self):
        return Quaternion(self._dq[4:])
    
    # -------------------------------------------------------------------------
    # Representation and indexing
    # -------------------------------------------------------------------------
    def __getitem__(self, key):
        return self._dq[key]

    def __repr__(self):
        return np.array_str(self._dq)


    # -------------------------------------------------------------------------
    # Arithmetic
    # -------------------------------------------------------------------------
    def __add__(self, other):
        if isinstance(other, DualQuaternion):
            return DualQuaternion(self.real + other.real, self.dual + other.dual)
        elif isinstance(other, np.ndarray):
            return DualQuaternion(Quaternion(self._dq[:4] + other[:4]),
                                  Quaternion(self._dq[4:] + other[4:]))
        else:
            raise TypeError("Unsupported addition type")

    def __sub__(self, other):
        if isinstance(other, DualQuaternion):
            return DualQuaternion(self.real - other.real, self.dual - other.dual)
        elif isinstance(other, np.ndarray):
            return DualQuaternion(Quaternion(self._dq[:4] - other[:4]),
                                  Quaternion(self._dq[4:] - other[4:]))
        else:
            raise TypeError("Unsupported subtraction type")

    def __mul__(self, other):
        if isinstance(other, DualQuaternion):
            # dual quaternion multiplication
            r = self.real * other.real
            d = self.real * other.dual + self.dual * other.real
            return DualQuaternion(r, d)
        elif np.isscalar(other):
            return DualQuaternion(self.real * other, self.dual * other)
        else:
            raise TypeError("Unsupported multiplication type")

    def __rmul__(self, other):
        if np.isscalar(other):
            # Scalar multiplication is commutative
            return self.__mul__(other)
        elif isinstance(other, DualQuaternion):
            # Reverse multiplication order for dual quaternions
            return other.__mul__(self)
        else:
            raise TypeError("Unsupported multiplication type")

    def __truediv__(self, other):
        if np.isscalar(other):
            return DualQuaternion(self.real / other, self.dual / other)
        else:
            raise TypeError("Division only supported by scalars")

    # -------------------------------------------------------------------------
    # Matrix multiplication
    # -------------------------------------------------------------------------
    def __rmatmul__(self, matrix):
        """8x8 matrix @ DualQuaternion."""
        assert isinstance(matrix, np.ndarray) and matrix.shape == (8, 8)
        out = matrix @ self._dq
        return DualQuaternion(Quaternion(out[:4]), Quaternion(out[4:]))

    # -------------------------------------------------------------------------
    # Dual quaternion operations
    # -------------------------------------------------------------------------
    def conj(self):
        return DualQuaternion(self.real.conj(), self.dual.conj())

    def swap(self):
        return DualQuaternion(self.dual, self.real)

    def dot(self, other):
        assert isinstance(other, DualQuaternion)
        r_dot = self.real.dot(other.real)
        d_dot = self.dual.dot(other.real) + self.real.dot(other.dual)
        return DualQuaternion(r_dot, d_dot)

    def cross(self, other):
        assert isinstance(other, DualQuaternion)
        r_cross = self.real.cross(other.real)
        d_cross = self.dual.cross(other.real) + self.real.cross(other.dual)
        return DualQuaternion(r_cross, d_cross)

    def norm_sq(self):
        return self.dot(self)
        
    def norm(self):
        return np.sqrt(self.norm_sq().real.q0)

    def normalize(self):
        real = self.real.normalize()
        dual = self.dual
        dq = DualPose(q_real=real, 
                      q_dual=dual - real * dual.dot(real))
        assert dq.isunit()
        return dq
    
    def vectorize(self):
        return DualVelocity(q_real=self.real.vectorize(), q_dual=self.dual.vectorize())
    
    # -------------------------------------------------------------------------
    # Matrix representations
    # -------------------------------------------------------------------------
    def left_mat(self):
        r_mat = self.real.left_mat()
        d_mat = self.dual.left_mat()
        return np.block([[r_mat, np.zeros((4, 4))],
                          [d_mat, r_mat]])

    def right_mat(self):
        r_mat = self.real.right_mat()
        d_mat = self.dual.right_mat()
        return np.block([[r_mat, np.zeros((4, 4))],
                          [d_mat, r_mat]])
        
    def cross_mat(self):
        r_mat = self.real.cross_mat()
        d_mat = self.dual.cross_mat()
        return np.block([[r_mat, np.zeros((4, 4))],
                          [d_mat, r_mat]])

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def isunit(self):
        return np.allclose(self.norm_sq().dq, np.array([1, 0, 0, 0, 0, 0, 0, 0]))

    def isvec(self):
        return self.real.isvec() and self.dual.isvec()    
        
class DualPose(DualQuaternion):
    """
    DualPose represents a dual quaternion pose.
    You must provide one of the following argument sets:
      - q_xy (unit quaternion) and r_xy_y (vector quaternion)
      - q_xy (unit quaternion) and r_xy_x (vector quaternion)
      - q_real (unit quaternion) and q_dual (quaternion)
    """
    def __init__(self, q_xy=None, r_xy_y=None, r_xy_x=None,
                 q_real=None, q_dual=None, dq=None):
        # Only one valid argument set should be provided
        arg_sets = [
            (q_xy is not None and r_xy_y is not None),
            (q_xy is not None and r_xy_x is not None),
            (q_real is not None and q_dual is not None),
            (dq is not None)
        ]
        if sum(arg_sets) != 1:
            raise ValueError('Provide exactly one valid argument set to DualPose')

        if q_xy is not None and r_xy_y is not None:
            assert q_xy.isunit(), "q_xy must be a unit quaternion (norm 1 rotation)"
            assert r_xy_y.isvec(), \
                "r_xy_y must be a pure-vector quaternion (zero scalar)."
            super().__init__(q_real=q_xy, q_dual=0.5 * r_xy_y * q_xy)
        elif q_xy is not None and r_xy_x is not None:
            assert q_xy.isunit(), "q_xy must be a unit quaternion (norm 1 rotation)"
            assert r_xy_x.isvec(), \
                "r_xy_x must be a pure-vector quaternion (zero scalar)."
            super().__init__(q_real=q_xy, q_dual=0.5 * q_xy * r_xy_x)
        elif q_real is not None and q_dual is not None:
            assert q_real.isunit(), "q_real must be a unit quaternion (norm 1 rotation)"
            super().__init__(q_real=q_real, q_dual=q_dual)
        elif dq is not None:
            assert Quaternion(dq[:4]).isunit(), \
                "dq[:4] must form a unit quaternion (rotation part)"
            super().__init__(dq=dq)
            
        # check unit dual quaternion constraints on the way out
        assert self.isunit(), 'Dual unit constraints are violated.'

    def __mul__(self, other):
        result = super().__mul__(other)
        # Type-preserving multiplication
        if isinstance(other, DualPose):
            return DualPose(dq=result.dq)
        else:
            return result
        
    def __rmul__(self, other):
        result = super().__rmul__(other)
        # Type-preserving multiplication
        if isinstance(other, DualPose):
            return DualPose(dq=result.dq)
        else:
            return result
    
    def conj(self):
        return DualPose(q_real=self.real.conj(), q_dual=self.dual.conj())
    
    @property
    def real(self):
        return UnitQuaternion(self._dq[:4])

    @property
    def dual(self):
        return Quaternion(self._dq[4:])
    
    @property
    def q_xy(self):
        return self.real
    
    @property
    def q_yx(self):
        return self.real.conj()
    
    @property
    def r_xy_x(self):
        return 2 * self.q_xy.conj() * self.dual
    
    @property
    def r_xy_y(self):
        return 2 * self.dual * self.q_xy.conj()
    
    @property
    def r_yx_x(self):
        return -1 * self.r_xy_x
    
    @property
    def r_yx_y(self):
        return -1 * self.r_xy_y
    
    def log(self, x_frame=True):
        if x_frame:
            return np.concatenate((self.real.log(), self.r_xy_x.qvec))
        else:
            return np.concatenate((self.real.log(), self.r_xy_y.qvec))
        

class DualVelocity(DualQuaternion):
    """
    DualVelocity represents a dual quaternion velocity.
    You must provide one of the following argument sets:
      - r_xz_z (vector quaternion), w_xy_z (vector quaternion), v_xy_z (vector quaternion)
      - q_real (vector quaternion), q_dual (vector quaternion), r_xz_z (vector quaternion)
    """
    def __init__(self, w_xy_z=None, v_xy_z=None, r_xz_z=None,
                 q_real=None, q_dual=None, dq=None):
        # Only one valid argument set should be provided
        arg_sets = [
            (w_xy_z is not None and v_xy_z is not None and r_xz_z is not None),
            (q_real is not None and q_dual is not None),
            (dq is not None)
        ]
        if sum(arg_sets) != 1:
            raise ValueError('Provide exactly one valid argument set to DualVelocity')

        if w_xy_z is not None and v_xy_z is not None and r_xz_z is not None:
            assert w_xy_z.isvec(), "w_xy_z must be a pure-vector quaternion."
            assert v_xy_z.isvec(), "v_xy_z must be a pure-vector quaternion."
            assert r_xz_z.isvec(), "r_xz_z must be a pure-vector quaternion."
            q_real=w_xy_z
            q_dual=v_xy_z - w_xy_z.cross(r_xz_z)
        elif dq is not None:
            q_real = Quaternion(dq[:4])
            q_dual = Quaternion(dq[4:])
            
        assert q_real.isvec(), "q_real must be a pure-vector quaternion (angular vel)."
        assert q_dual.isvec(), \
            "q_dual must be a pure-vector quaternion (linear vel minus cross term)."
        super().__init__(q_real=q_real, q_dual=q_dual)
            
        # check vector constraints on the way out
        assert self.isvec(), 'Dual velocity is not a dual vector.'

    def __add__(self, other):
        result = super().__add__(other)
        # Type-preserving multiplication
        if isinstance(other, DualVelocity):
            return DualVelocity(dq=result.dq)
        else:
            return result
        
    def __sub__(self, other):
        result = super().__sub__(other)
        # Type-preserving multiplication
        if isinstance(other, DualVelocity):
            return DualVelocity(dq=result.dq)
        else:
            return result
    
    @property
    def w_xy_z(self):
        return self.real
    
    @property
    def w_yx_z(self):
        return self.w_xy_z.conj()
    
    def v_xy_z(self, r_xz_z):
        assert r_xz_z.isvec()
        out = self.dual + self.w_xy_z.cross(r_xz_z)
        assert out.isvec()
        return out
    
    def v_yx_z(self, r_xz_z):
        return self.v_xy_z(r_xz_z).conj()
    
class DualForce(DualQuaternion):
    def __init__(self, force=None, torque=None, dq=None):
        # Only one valid argument set should be provided
        arg_sets = [
            (force is not None and torque is not None),
            (dq is not None)
        ]
        if sum(arg_sets) != 1:
            raise ValueError('Provide exactly one valid argument set to DualForce')
            
        if dq is not None:
            force = Quaternion(dq[:4])
            torque = Quaternion(dq[4:])
        assert force.isvec(), "force must be a pure-vector quaternion."
        assert torque.isvec(), "torque must be a pure-vector quaternion."
        super().__init__(q_real=force, q_dual=torque)

        # check vector constraints on the way out
        assert self.isvec(), 'Dual force is not a dual vector.'

    @property
    def force(self):
        return self.real

    @property
    def torque(self):
        return self.dual

def qone():
    return UnitQuaternion(np.array([1, 0, 0, 0]))

def qzero():
    return Quaternion(np.zeros(4))

def dqone():
    return DualPose(qone(), qzero())

def dqzero():
    return DualQuaternion(qzero(), qzero())

def I_conj():
    return np.diag([1, -1, -1, -1])
    
def build_M(m, J):
    """
    Construct 8x8 block matrix:
        [[0_{4x4}, diag(1,m,m,m)],
         [1, 0_{1x7}],
         [0_{3x1}, J_{3x3}, 0_{3x4}]]

    Args:
        m (scalar): positive scale factor.
        J (array-like): 3x3 positive-definite matrix.

    Returns:
        ndarray: 8x8 matrix.

    Raises:
        AssertionError: if inputs invalid.
    """
    # Validate m
    assert np.isscalar(m) and m > 0, "m must be a scalar"

    # Validate J
    assert J.shape == (3, 3), "J must be shape (3,3)"
    assert np.allclose(J, J.T, atol=1e-12), "J must be symmetric"
    assert np.all(np.linalg.eigvals(J) > 0), 'J must be PD'

    return np.block([
        [np.zeros((4, 1)), np.zeros((4, 3)), np.diag([1.0, m, m, m])],
        [np.array([[1.0]]), np.zeros((1, 3)), np.zeros((1, 4))],
        [np.zeros((3, 1)), J, np.zeros((3, 4))]
    ])

def kinematics(dq_xy, dw_xy_y=None, dw_xy_x=None):
    """
    Dual quaternion kinematics:
      - If velocity is expressed in the 'y' frame (right-multiplied):  0.5 * dw_xy_y * dq_xy
      - If velocity is expressed in the 'x' frame (left-multiplied):   0.5 * dq_xy * dw_xy_x

    Exactly one of dw_xy_y or dw_xy_x must be provided.
    """
    vy = dw_xy_y is not None
    vx = dw_xy_x is not None
    if (vy + vx) != 1:
        raise ValueError("Provide exactly one of dw_xy_y or dw_xy_x.")
    if vy:
        return 0.5 * (dw_xy_y * dq_xy)
    else:
        return 0.5 * (dq_xy * dw_xy_x)

def dynamics(dw_xy_x, df_x, M_x, M_x_inv=None):
    assert isinstance(df_x, DualForce)
    assert M_x.shape == (8, 8)
    
    if M_x_inv is None:
        M_x_inv = np.linalg.inv(M_x)
    else:
        assert np.allclose(M_x_inv @ M_x, np.eye(M_x.shape[0]))
    out = M_x_inv @ (df_x - dw_xy_x.cross(M_x @ dw_xy_x))
    assert out.isvec(), 'Dual velocity time derivative must be a vector dual quaternion'
    return out

def relative_dynamics(dq_tc, dw_tc_c, dw_cj_c, dot_dw_cj_c, dot_dw_tj_t):
    assert isinstance(dw_cj_c, DualVelocity)
    assert isinstance(dot_dw_cj_c, DualQuaternion)
    assert isinstance(dot_dw_tj_t, DualQuaternion)
    dot_dw_tc_c = (dq_tc * dot_dw_tj_t * dq_tc.conj()
                   + dw_tc_c.cross(dw_cj_c) - dot_dw_cj_c)
    assert dot_dw_tc_c.isvec(), \
        'Dual velocity derivative must be a vector dual quaternion.'
    return dot_dw_tc_c

def rand_qunit(low=-1, high=1):
    u = np.random.uniform(low=low, high=high, size=4)
    return Quaternion(u).normalize()

def rand_qvec(low=-1, high=1):
    v = np.random.uniform(low=low, high=high, size=3)
    return Quaternion(np.concatenate([np.array([0]), v]))

def rand_dqvec(w_low=-1, w_high=1, v_low=-1, v_high=1):
    # random dual vector
    w = rand_qvec(low=w_low, high=w_high)
    v = rand_qvec(low=v_low, high=v_high)
    return DualQuaternion(w, v)

def rand_dpose(v_low=-1, v_high=1):
    # random dual quaternion dq_x_y
    return DualPose(q_xy=rand_qunit(), r_xy_x=rand_qvec(low=v_low, high=v_high))

def state2array(x):
    return np.concatenate((x[0].dq, x[1].dq))

def state2dq(x):
    return x[0], x[1]

def array2dq(x, correct=False):
    if correct:
        dq = DualQuaternion(dq=x[:8]).normalize()
    else:
        dq = DualQuaternion(dq=x[:8])
            
    if x.size == 8:
        return dq       
    
    if correct:
        dw = DualQuaternion(dq=x[8:]).vectorize()
    else:
        dw = DualQuaternion(dq=x[8:])

    return dq, dw

def dq2array(dq, dw=None):
    assert isinstance(dq, DualQuaternion)
    if dw is None:
        return dq.dq, None
    else:
        assert isinstance(dw, DualQuaternion)
        return np.concatenate((dq.dq, dw.dq))
    
def dq2state(dq, dw):
    assert dq.isunit()
    assert dw.isvec()
    dq = DualPose(dq=dq.dq)
    dw = DualVelocity(dq=dw.dq)
    return np.array([dq, dw])

def log(x, x_frame=True):
    # transform DQ to tangent space
    if x_frame:
        dq_xy, dw_xy_x = x[0], x[1]
        assert isinstance(dq_xy, DualPose)
        assert isinstance(dw_xy_x, DualVelocity)
        r_xy_x = dq_xy.r_xy_x
        return np.concatenate((dq_xy.q_xy.log(),
                                r_xy_x.qvec,
                                dw_xy_x.w_xy_z.qvec,
                                dw_xy_x.v_xy_z(r_xz_z=qzero()).qvec))
    else:
        dq_xy, dw_xy_y = x[0], x[1]
        assert isinstance(dq_xy, DualPose)
        assert isinstance(dw_xy_y, DualVelocity)
        r_xy_y = dq_xy.r_xy_y
        return np.concatenate((dq_xy.q_xy.log(),
                                r_xy_y.qvec,
                                dw_xy_y.w_xy_z.qvec,
                                dw_xy_y.v_xy_z(r_xz_z=r_xy_y).qvec))
        
def exp(x_err):
    # transform error state to DQ state
    q_tc = UnitQuaternion(x_err[:3])
    r_tc_c = Quaternion(np.concatenate((np.zeros(1), x_err[3:6])))
    if x_err.size == 6:
        dq_tc = DualPose(q_xy=q_tc, r_xy_y=r_tc_c)
        return dq_tc
    
    elif x_err.size == 12:
        w_tc_c = Quaternion(np.concatenate((np.zeros(1), x_err[6:9])))
        v_tc_c = Quaternion(np.concatenate((np.zeros(1), x_err[9:12])))
        
        dq_tc = DualPose(q_xy=q_tc, r_xy_y=r_tc_c)
        dw_tc_c = DualVelocity(w_xy_z=w_tc_c, v_xy_z=v_tc_c, r_xz_z=r_tc_c)
        return np.array([dq_tc, dw_tc_c])

def retract(x_manifold, x_err):
    dq, dw = x_manifold[0], x_manifold[1]
    
    delta_q_tc = UnitQuaternion(x_err[:3])
    delta_r_tc_c = Quaternion(np.concatenate((np.zeros(1), x_err[3:6])))
    delta_w_tc_c = Quaternion(np.concatenate((np.zeros(1), x_err[6:9])))
    delta_v_tc_c = Quaternion(np.concatenate((np.zeros(1), x_err[9:12])))
        
    dq_delta = DualPose(q_xy=delta_q_tc, r_xy_y=delta_r_tc_c)
    
    # perturb pose on manifold, velocities additively
    dq_perturb = dq * dq_delta
    dw_perturb = DualVelocity(w_xy_z=dw.w_xy_z + delta_w_tc_c,
                              v_xy_z=dw.v_xy_z(dq.r_xy_y) + delta_v_tc_c,
                              r_xz_z=dq.r_xy_y + delta_r_tc_c)
    
    return np.array([dq_perturb, dw_perturb])

def lift(x_mean, x_point):
    if isinstance(x_mean, DualPose) and isinstance(x_point, DualPose):
        dq_delta = x_mean.conj() * x_point
        out = np.concatenate([dq_delta.q_xy.log(),
                        dq_delta.r_xy_y.qvec])
        return out
    
    else:
        dq_point = x_point[0]
        dw_point = x_point[1]
        xw_point = np.concatenate((dw_point.w_xy_z.qvec,
                                dw_point.v_xy_z(dq_point.r_xy_y).qvec))
        
        dq_mean = x_mean[0]
        dw_mean = x_mean[1]
        xw_mean = np.concatenate((dw_mean.w_xy_z.qvec,
                                dw_mean.v_xy_z(dq_mean.r_xy_y).qvec))
        
        dq_delta = x_mean[0].conj() * x_point[0]
        xw_delta = xw_point - xw_mean
        out = np.concatenate([dq_delta.q_xy.log(),
                        dq_delta.r_xy_y.qvec,
                        xw_delta])
        return out