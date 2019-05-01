%% Burgers steady state optimal control %%
% optimal control optimization for steady burgers equation
% aided by time stepping simulation for verification

% tested matlab built-in nonlinear optimization algorithms: 
% sqp, interior-point and active-set
% tested cases where Hessian and gradients are provided versus
% approximation


% optimizing variables: y - steady state profile, u - control
clear all; close all
n.x = 20;
n.y = n.x - 1;
n.u = n.x + 1;
n.dx = 1/n.x;
n.px = linspace(0,1,n.u)';

Matrices.M = n.dx/6*(diag(ones(n.y-1,1),1)+diag(ones(n.y-1,1),-1)+4*diag(ones(n.y,1)));
Matrices.A = 1/n.dx*(-diag(ones(n.y-1,1),1)-diag(ones(n.y-1,1),-1)+2*diag(ones(n.y,1)));
Matrices.B = -n.dx/6*(diag(ones(n.y+1,1),-1)+4*diag(ones(n.u,1))+diag(ones(n.y+1,1),1));
Matrices.B = Matrices.B(2:end-1,:);
Matrices.Q = diag(ones(n.u-1,1),1)+diag(ones(n.u-1,1),-1)+4*diag(ones(n.u,1));
Matrices.Q([1,end],[1,end]) = 2; Matrices.Q = n.dx/6*Matrices.Q;

% control and simulation parameters
parameter.w = 0.01;              % penalty for control
parameter.v = 0.1;               % diffusivity
parameter.maxiter = 1000;        % max iteration number
parameter.dt = 0.01;             % time step
parameter.tol = 1e-6;            % tolerance for steady state
parameter.DT = 0.1;              % interval for plotting
parameter.opt = 0;               % plotting toggle: 1 to plot

forcing.r = -0.1*ones(n.y,1);    % fixed forcing
u0 = zeros(n.u,1);            % u profile guess
y_hat = -n.px(2:end-1).*(n.px(2:end-1)-1);    % target y profile

y.initial = zeros(n.y,1);        % initial y profile
forcing.u = u0;
y0 = burgers_sim(n,parameter,Matrices,forcing,y);  % feasible y
x0 = [y0;u0];                    % feasible initial guess


% change simulation parameters for faster plotting
parameter.opt = 1; 
parameter.tol = 5e-5;
%% SQP with approximated gradients %%
options = optimoptions(@fmincon,'Algorithm','sqp','Display','off');
message = 'SQP method without gradients...\n';
[~,OUTPUT] = verify_func(options,x0,n,Matrices,parameter,y_hat,forcing,message);

%% SQP with analytical gradients %%
clear options
options = optimoptions(@fmincon,'Algorithm','sqp','Display','off');
options = optimoptions(options,'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true);
message = 'SQP method with gradients...\n';
[X,~] = verify_func(options,x0,n,Matrices,parameter,y_hat,forcing,message);

%% Interior-point without Hessian %%
clear options
options = optimoptions(@fmincon,'Algorithm','interior-point','OptimalityTolerance',1e-5,...
        'Display','off','SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true);
message = 'Interior-point method without Hessian...\n';
verify_func(options,x0,n,Matrices,parameter,y_hat,forcing,message);

%% Interior-point with Hessian %%
x0 = X*.9;
clear options
options = optimoptions(@fmincon,'Algorithm','interior-point','OptimalityTolerance',1e-5,...
        'Display','off','SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,...
        'HessianFcn',@(x,lambda)hessinterior(x,lambda,n,Matrices,parameter));
% options = optimoptions(@fmincon,'Algorithm','interior-point',...
%         'Display','off','SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,...
%         'SubproblemAlgorithm','cg',...
%         'HessianMultiplyFcn',@(x,lambda,z)hessmulti(x,lambda, z, n,Matrices,parameter));
message = 'Interior-point method with Hessian (starting close to solution)...\n';
verify_func(options,x0,n,Matrices,parameter,y_hat,forcing,message);

%% Active set %%
x0 = [y0;u0];
clear options
options = optimoptions(@fmincon,'Algorithm','active-set',...
        'Display','off','SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true);
message = 'Active set method with gradients...\n';
verify_func(options,x0,n,Matrices,parameter,y_hat,forcing,message);


%% SQP with line search %%
clear options
x0 = [y0;u0];
fprintf('SQP method with line search globalization...\n')
options.iprint = 0; % default = 0
options.fid    = 1; % file identifier, default = 1
options.maxit  = 400; %  maximim number of iterations, default = 100
options.tol    = 1e-6; % residual stopping tolerance, default = 1.e-8

l1 = zeros(n.y,1);
l2 = [];

[X,l1,l2,iflag,hist]= my_sqp_l1(@(x)objfun(x,n,Matrices,parameter,y_hat),...
    @(x)confun(x,n,Matrices,forcing.r,parameter),x0,l1,l2,options);

fprintf( ' my_sqp_l1 returned with iflag = %d \n', iflag )
fprintf( ' Iteration history \n' )
fprintf( '  k      f_k    ||hk,(g_k)_+||_1   optim      rho_k     stepsize_k \n')
for iter = 0:size(hist,1)-1
    fprintf( ' %2d   %10.4e   %10.4e    %10.4e  %10.3e  %10.3e \n', iter+1, hist(iter+1,2:6) )
end

% verify with simulation
fprintf('Running simulation for verfication...\n\n')
y.initial = x0(1:n.y);
y.target  = [0; y_hat; 0];
y.steady  = [0; X(1:n.y); 0];
forcing.u = X(n.y+1:end);
figure
burgers_sim(n,parameter,Matrices,forcing,y)


%% helper functions for optimization %%
function [X, OUTPUT] = verify_func(options,x0,n,Matrices,parameter,y_hat,forcing,message)
% standard optimization and verification function
% uses specified algorithm to solve nonlinear optimization and runs
% simulation to verify results

% returns optimal X = [y; u]
[X,FVAL,~,OUTPUT] = fmincon(@(x)objfun(x,n,Matrices,parameter,y_hat),x0,[],[],[],[],[],[],... 
   @(x)confun(x,n,Matrices,forcing.r,parameter),options);
fprintf(message)
fprintf(['Algorithm : ',OUTPUT.algorithm,'\t\t\t Iteration taken: %i\n'],OUTPUT.iterations)
fprintf('Objective value: %5.3e \t Constrain violation: %5.3e\n', FVAL, OUTPUT.constrviolation)
fprintf('Step: %5.3e \t\t Optimality: %5.3e\n', OUTPUT.stepsize, OUTPUT.firstorderopt)
% verify with simulation
fprintf('Running simulation for verfication...\n\n')
y.initial = x0(1:n.y);
y.target  = [0; y_hat; 0];
y.steady  = [0; X(1:n.y); 0];
forcing.u = X(n.y+1:end);
figure
burgers_sim(n,parameter,Matrices,forcing,y);
end

function [c,ceq,DC,DCeq] = confun(x,n,Matrices,r,parameter)
% return: c     - inequality constraint
%         ceq   - equality constraint
%         DC    - Jacobian of inequality constraint
%         DCeq  - Jacobian of equality constraint
y = x(1:n.y);
u = x(n.y+1:end);
ceq = parameter.v*Matrices.A*y + ...
    Nh(y,n.y) + Matrices.B*u - Matrices.M*r;
c = [];  
if nargout > 2
    DC= [];
    DCeq = [parameter.v*Matrices.A' + Nh_dash(y)';...
            Matrices.B' ];
end
end

function [f,gradf] = objfun(x,n,Matrices,parameter,y_hat)
% return: f     - objective function value
%         gradf - gradient of objective function
y = x(1:n.y);
u = x(n.y+1:end);
f = 1/2*y'*Matrices.M*y - y_hat'*Matrices.M*y + ...
    parameter.w/2*u'*Matrices.Q*u;
if nargout  > 1
    gradf = [Matrices.M*y - Matrices.M*y_hat; ...
        parameter.w*Matrices.Q*u];
end
end


function h = hessinterior(x,lambda,n,Matrices,parameter)
% provide hessian of lagrangian for interior-point method
N_dash2 = diag(2*[0;lambda.eqnonlin(1:end-1)]-2*[lambda.eqnonlin(2:end);0]) + ...
    diag(lambda.eqnonlin(1:end-1)-lambda.eqnonlin(2:end),1) + ...
    diag(lambda.eqnonlin(1:end-1)-lambda.eqnonlin(2:end),-1);
N_dash2 = N_dash2/6;
h = [Matrices.M + N_dash2, zeros(n.y,n.u); ...
    zeros(n.u,n.y), parameter.w*Matrices.Q];
end


function W = hessmulti(x,lambda, z, n,Matrices,parameter)
% provide multilied hessian of lagrangian for interior-point method
v = z(1:n.y); 
w = z(n.y+1:end);
W = [Matrices.M*v + Nh_dash(v)'*lambda.eqnonlin; ...
    parameter.w*Matrices.Q*w];
end
