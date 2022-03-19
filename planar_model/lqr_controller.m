syms rB rW mB mW mC l g t
syms theta(t) thetad(t) phi(t) phid(t) tau(t)

% Computing Lagrangian
TB = mB * rB^2 * phid^2;
UB = 0;

TW = 0.5*mW*(rB^2*phid^2 + 2*rB*(rB+rW)*phid*thetad*cos(theta)...
        +(rB+rW)^2*phid^2+rW^2*phid^2);
UW = mW*g*(rB+rW)*cos(theta);

TC = 0.5*mC*(rB*phid^2 + 2*l*rB*phid*thetad*cos(theta)+l^2*thetad^2);
UC = mC*g*l*cos(theta);

L = TB + TW + TC - UB - UW - UC;

% For equations of motion
x = {theta thetad phi phid};

Q_i = {0 0}; Q_e = {-tau*(rB/rW) tau*(rB/rW)};
R = 0;
par = {rB rW mB mW mC l g};

VF = EulerLagrange(L, x, Q_i, Q_e, R, par);

% Linearise around fixed point
xd = simplify(VF, 'Steps', 100); % [thetad, thetadd, phid, phidd]
x = cell2sym(x); % [theta, thetad, phi, phid]

A = jacobian(xd, x);
B = jacobian(xd, tau);

A = subs(A, {theta thetad phi phid tau}, {0 0 0 0 0});
B = subs(B, {theta thetad phi phid tau}, {0 0 0 0 0});

% Substitute in params
A = subs(A, {mB mW mC rB rW l g}, {1 0.25 0.5 0.06 0.035 0.335 9.81});
B = subs(B, {mB mW mC rB rW l g}, {1 0.25 0.5 0.06 0.035 0.335 9.81});
%A = subs(A, {mB mW mC rB rW l g}, {1000 250 500 60 35 335 9810});
%B = subs(B, {mB mW mC rB rW l g}, {1000 250 500 60 35 335 9810});
A = double(A);
B = double(B);

% Bryson's Rule
max_ball_angle = 360;
max_ball_angular_velocity = 360; 
max_rod_angle = 5;
max_rod_angular_velocity = 38;
max_torque = 28000;

% 1. have rod angle and rod angular velocity at same decent value
% 2. tweak torque until max duration
% 3. increase angular velocity until barely oscillate
% 4. decrease ball angle and ball angular velocity

q1 = 1/deg2rad(max_ball_angle)^2;
q2 = 1/deg2rad(max_ball_angular_velocity)^2;
q3 = 1/deg2rad(max_rod_angle)^2;
q4 = 1/deg2rad(max_rod_angular_velocity)^2;

r1 = 1/(max_torque)^2;

Q = [q1, 0 , 0,  0;
     0, q2,  0,  0;
     0,  0, q3,  0;
     0,  0,  0, q4]

R = r1*eye(1)

%rank(ctrb(A, B))

K = lqr(A, B, Q, R)
writematrix(K, 'lqr_gains.txt');