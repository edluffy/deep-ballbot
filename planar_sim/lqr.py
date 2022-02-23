import numpy as np

from sympy import sin, cos, Matrix, Function, symbols, solve, simplify, Eq, latex
from sympy.calculus.euler import euler_equations
from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols

from scipy import linalg

class LQR:
    def __init__(self):
        self.A, self.B = self.get_linearised_dynamics()

    def get_linearised_dynamics(self):
        rB, rW, mB, mW, mC, l, g, t = symbols('rB rW mB mW mC l g t')
        theta, phi, tau = dynamicsymbols('theta phi tau')
        theta_dot, phi_dot = dynamicsymbols('theta phi', 1)

        TB = mB * rB**2 * phi_dot**2
        UB = 0

        TW = 0.5*mW*(rB**2*phi_dot**2 + 2*rB*(rB+rW)*phi_dot*theta_dot*cos(theta)
                +(rB+rW)**2*phi_dot**2+rW**2*phi_dot**2)
        UW = mW*g*(rB+rW)*cos(theta)

        TC = 0.5*mC*(rB*phi_dot**2 + 2*l*rB*phi_dot*theta_dot*cos(theta)+l**2*theta_dot**2)
        UC = mC*g*l*cos(theta)

        L = TB + TW + TC - UB - UW - UC

        euler_expr = euler_equations(L, funcs=[theta, phi])
        eq1 = euler_expr[0].lhs# - -tau*(rB/rW)
        eq2 = euler_expr[1].lhs# - -tau*(rB/rW)

        M = Matrix([
            [eq1.coeff(theta.diff().diff()), eq1.coeff(phi.diff().diff())],
            [eq2.coeff(theta.diff().diff()), eq2.coeff(phi.diff().diff())]
        ])

        #print(simplify(eq1).coeff(theta.diff().diff()))
        print(latex(eq1))
        print('\n')
        print(latex(simplify(eq1)))

        theta_dot_dot = solve(eq1, theta.diff().diff())[0]
        phi_dot_dot = solve(eq2, phi.diff().diff())[0]

        x1 = theta
        x2 = phi
        x3 = theta_dot
        x4 = phi_dot

        xdot = Matrix([x3, x4, theta_dot_dot, phi_dot_dot])

        A = xdot.jacobian([x1, x2, x3, x4])
        B = xdot.jacobian([tau])

        fixed_point = [(theta, 0), (phi, 0), (theta_dot, 0), (phi_dot, 0)]
        A = A.subs(fixed_point)
        B = B.subs(fixed_point)


        params = {
                mB: 1000,
                mW: 250,
                mC: 500,
                rB: 60,
                rW: 35,
                l: 335,
                g: 981
        }

        A = np.array(A.evalf(subs=params).doit(), dtype=float)
        B = np.array(B.evalf(subs=params).doit(), dtype=float)

        return A, B

    def policy(self, state):
        R = 1*np.eye(1)
        Q = 1*np.eye(4)

        P = linalg.solve_continuous_are(self.A, self.B, Q, R)

        K = np.linalg.inv(R) @ (self.B.T @ P)

        u = (-K @ state)[0]

        return u

test = LQR()
