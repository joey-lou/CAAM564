function J = Nh_dash(y)
% returns Jacobian of Nh
J = diag(-2*y(1:end-1)-y(2:end),-1) + ...
    diag(y(1:end-1)+2*y(2:end),1) + ...
    diag([y(2:end);0]-[0;y(1:end-1)]);
J = J/6;
end