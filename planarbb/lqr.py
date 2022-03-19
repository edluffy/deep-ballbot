import numpy as np

from sympy import sin, cos, Matrix, Function, symbols, solve, simplify, Eq, latex, expand, collect
from sympy.calculus.euler import euler_equations
from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols

from scipy import linalg

rB, rW, mB, mW, mC, l, g, t = symbols('rB rW mB mW mC l g t')
theta, phi, tau = dynamicsymbols('theta phi tau')
theta_dot, phi_dot = dynamicsymbols('theta phi', 1)

class LQR:
    def __init__(self):
        self.K = self.get_gains()

        #L = self.compute_lagrangian()
        #q_dot_dot = self.compute_motion_eq(L)
        #A, B = self.get_state_space(q_dot_dot)

        ## Substitute in params
        #params = {
        #        mB: 1000,
        #        mW: 250,
        #        mC: 500,
        #        rB: 60,
        #        rW: 35,
        #        l: 335,
        #        g: 981
        #}

        #self.A = np.array(A.evalf(subs=params).doit(), dtype=float)
        #self.B = np.array(B.evalf(subs=params).doit(), dtype=float)

        #print(self.A)
        #print('\n')
        #print(self.B)

    def get_gains(self):
        with open('lqr_gains.txt') as f:
            line = f.readline().split(',')
            K = np.array([float(k) for k in line])
        return K

    def compute_lagrangian(self):
        TB = mB * rB**2 * phi_dot**2
        UB = 0

        TW = 0.5*mW*(rB**2*phi_dot**2 + 2*rB*(rB+rW)*phi_dot*theta_dot*cos(theta)
                +(rB+rW)**2*phi_dot**2+rW**2*phi_dot**2)
        UW = mW*g*(rB+rW)*cos(theta)

        TC = 0.5*mC*(rB*phi_dot**2 + 2*l*rB*phi_dot*theta_dot*cos(theta)+l**2*theta_dot**2)
        UC = mC*g*l*cos(theta)

        L = TB + TW + TC - UB - UW - UC
        return L

    def compute_motion_eq(self, L):
        euler_expr = euler_equations(L, funcs=[theta, phi])
        eq1 = euler_expr[0].lhs.expand() - tau*(rB/rW)
        eq2 = euler_expr[1].lhs.expand() #- tau*(rB/rW+1)

        M = simplify(Matrix([
            [eq1.coeff(theta.diff().diff()), eq1.coeff(phi.diff().diff())],
            [eq2.coeff(theta.diff().diff()), eq2.coeff(phi.diff().diff())]
        ]))

        F = simplify(Matrix([eq1, eq2]) - M*Matrix([theta.diff().diff(), phi.diff().diff()]))

        q_dot_dot = M.inv()*F
        return q_dot_dot

    def get_state_space(self, q_dot_dot):
        theta_dot_dot, phi_dot_dot = q_dot_dot

        x = Matrix([theta, phi, theta_dot, phi_dot])
        x_dot = Matrix([theta_dot, phi_dot, theta_dot_dot, phi_dot_dot])

        # Linearise around fixed point x: {0, 0, 0, 0}
        A = x_dot.jacobian(x)
        B = x_dot.jacobian([tau])

        fixed_point = [(theta, 0), (phi, 0), (theta_dot, 0), (phi_dot, 0)]
        A = A.subs(fixed_point)
        B = B.subs(fixed_point)

        #print(latex(simplify(A)))
        #print('\n')
        #print(latex(simplify(B)))

        return A, B

    def policy(self, state):
        #R = 0.1*np.eye(1)
        ##Q = 5*np.eye(4)
        #Q = np.array([
        #    [10, 0, 0, 0],
        #    [0, 60, 0, 0],
        #    [0, 0, 10, 0],
        #    [0, 0, 0, 10]
        #])

        #P = linalg.solve_continuous_are(self.A, self.B, Q, R)
        #K = np.linalg.inv(R) @ (self.B.T @ P)

        #K = 1e8 * np.array([-3.136, -1.5131, 0, 0])
        #K = 1e8 * np.array([-3.1396, -1.5172, 0, -0.0002])

        u = (-self.K @ state)

        return u
