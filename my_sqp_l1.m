function [x,y,z,iflag,hist] = my_sqp_l1(objective,constraint,x,y,z,options);
%
% Solves the NLP
%
%       min  f(x)
%       s.t. h(x) = 0, 
%            g(x) <= 0
%
% using an SQP method with line seaech and l1 merit function
%
% inputs:
%  objective   [f,grad_f] = objective(x) evaluate objective function
%              and its gradient (columns vector)
%  constraint  [g,h,J_g,J_h] = constraint(x)   evaluate inequality (g(x) <= 0)
%              and equality constrains (h(x) = 0), and their Jacobians
%  x0       vector initial primal variables x 
%  y0       vector initial Lagrange multipliers y for equality constraints
%  z0       vector initial Lagrange multipliers z for inequality constraints
%
%  options  algorithm options
%             options.iprint   print if ~=0, default = 0
%             options.fid      file identifier, default = 1
%             options.maxit    maximim number of iterations, default = 100
%             options.tol      stopping tolerance, default = 1.e-8
%             if  my_sqp_l1  is called with options =[], then defauls
%                   are used.
%         
% outputs:
%  x	    solution to the QP   (only if flag==0)
%  y	    Lagrange multipliers for equality constr. (only if flag==0)
%  z	    Lagrange multipliers for inequality constr. (only if flag==0)
%
%  iflag	0 if optimal solution was found
%           1 no optimal solution found in maxit iterations
%  hist     iteration history. the (iter+1)st row contains
%           [iter, norm(AE*x-bE), norm(AI*x+s-bI), ...
%             norm(H*x+AE'*y+AI'*z+c), s'*z/mI, stepsize]
%
%
%  AUTHOR:  Matthias Heinkenschloss
%           Department of Computational and Applied Mathematics
%           Rice University
%           March 24, 2019
%           Last modified March 25, 2019
%
%

% set parameters for Armijo step-size
sigma = 1.e-4; beta = 0.5; 

% set algorithm options
iprint  = 0;
fid     = 1;
maxit   = 100;
tol     = 1.e-8;
if( ~isempty(options) )
    if( isfield(options,'iprint') ); iprint = options.iprint; end
    if( isfield(options,'fid') );    fid    = options.fid; end
    if( isfield(options,'maxit') );  maxit  = options.maxit; end
    if( isfield(options,'tol') );    tol    = options.tol; end
end

QP_options = optimset('Display', 'off', 'LargeScale', 'off');
%QP_options = optimset('Display', 'off');

iter          = 0;
[f,grad_f]    = feval(objective,x);
[g,h,J_g,J_h] = feval(constraint,x);
H             = eye(length(x));
Optimality    = norm( [grad_f+J_h*y; h; min(-g,z)] );

while (Optimality>tol) && (iter<maxit)
    [sx,fval,exitflag,output,lambda] = quadprog(H,grad_f,J_g,-g,...
                                                J_h',-h,[],[],[],QP_options);
    z_s        = lambda.ineqlin;
    y_s        = lambda.eqlin;
    rho        = max(norm(z_s,inf),norm(y_s,inf));
    % Line search (step size is t)
    P1_der     = -sx'*H*sx;
    t          = 1; 
    P1         = f + rho*( sum(max(g,zeros(size(g)))) + norm(h,1) );
    f_t        = feval(objective,x+sx);
    [g_t,h_t]  = feval(constraint,x+sx);
    P1_t       = f_t + rho * ( sum(max(g_t,zeros(size(g)))) + norm(h_t,1) );
    while (P1_t > P1  + sigma*t*P1_der)
        t         = beta*t;
        f_t       = feval(objective,x+t*sx);
        [g_t,h_t] = feval(constraint,x+t*sx);
        P1_t      = f_t + rho * ( sum(max(g_t,zeros(size(g)))) + norm(h_t,1) );
    end
    
    hist(iter+1,:) = [ iter, f, sum(max(g,zeros(size(g)))) + norm(h,1), ...
                       Optimality, rho, t ];
    if (iprint)
        fprintf(fid,'Iteration %d \n',iter);
        fprintf(fid,'  Objective function   = %12.6e \n',f );
        fprintf(fid,'  Constraint violation = %12.6e \n',sum(max(g_t,zeros(size(g)))) + norm(h_t,1) );
        fprintf(fid,'  Optimality           = %12.6e \n',Optimality );
        fprintf(fid,'  Penalty parameter    = %12.6e \n',rho );
        fprintf(fid,'  step size            = %12.6e \n',t );
    end
            
    x_s                   = x+t*sx;
    [f_s,grad_f_s]        = feval(objective,x_s);
    [g_s,h_s,J_g_s,J_h_s] = feval(constraint,x_s);
    % Update Hessian approximation using damped BFGS
    s = x_s-x; 
    v = grad_f_s-grad_f+(J_h_s-J_h)*y;
    vs  = v'*s; 
    Hs  = H*s; 
    sHs = s'*Hs;
    if vs >= 0.2*sHs
        theta = 1;
    else
        theta = 0.8*sHs/(sHs-vs);
    end
    v          = theta*v+(1-theta)*Hs;
    H          = H-(Hs)*(Hs)'/sHs+v*v'/(v'*s);
    f          = f_s; grad_f = grad_f_s;g = g_s;J_g = J_g_s;h = h_s;J_h = J_h_s;x = x_s;z = z_s;y = y_s;
    Optimality = norm( [grad_f+J_h*y; h; min(-g,z)] );
    iter       = iter+1;
end

if norm(Optimality,inf) <= tol
    iflag = 0;
else
    iflag = 1;
end
