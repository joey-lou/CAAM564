function STATE = burgers_sim(n,parameter,Matrices,forcing,y)
% solve 1D burger's equation using FEM
% piece-wise linear basis for x
% crank-nicolson and adam-bashforth for t
% dirichlet boundary conditions enforced

% input: n.x        - grid count
%        n.y        - point count for y
%        n.u        - point count for u
%        n.dx       - grid size
%        n.px       - x grid for plotting
%        Matrices.A - FEM diffusion integration matrix
%        Matrices.B - FEM control integration matrix
%        Matrices.M - FEM fixed integration matrix
%        forcing.u  - control
%        forcing.r  - fixed forcing
%        y.inital   - initial y profile
%        y.target   - target y profile
%        y.steady   - steady state y profile

y.current = y.initial;

Adv.current = Nh(y.current,n.y);
Adv.previous = Adv.current;

LHS = Matrices.M/parameter.dt + parameter.v*Matrices.A/2;
LHS = inv(LHS);
MrBu = Matrices.M*forcing.r-Matrices.B*forcing.u;

iter = 0;
while iter<parameter.maxiter
    RHS = MrBu + 1/2*Adv.previous - 3/2*Adv.current +...
        (Matrices.M/parameter.dt-parameter.v*Matrices.A/2)*y.current;
    temp = LHS*RHS;
    y.previous = y.current;
    y.current = temp;
    Adv.previous = Adv.current;
    Adv.current = Nh(y.current,n.y);
    iter = iter + 1;
    err = norm(y.previous-y.current);
    if mod(iter,parameter.DT/parameter.dt)==0
        if err<parameter.tol
            STATE = y.current;
            break
        end
        if parameter.opt == 1
            plot(n.px,[0;y.current;0],n.px,forcing.u,'--',n.px(2:end-1),...
                forcing.r,'--',n.px,y.target,'*',n.px,y.steady,'o','linewidth',1.5)
            title(['t = ',num2str(iter*parameter.dt,'%4.2f'),...
                ', \delta y = ',num2str(err,'%4.2e')])
            xlabel('x')
            ylabel('y')
            legend('y','control u','forcing r','target y','steady y')
            pause(0.01)
        end
    end
    
end
end