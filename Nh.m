function x = Nh(y,ny)
% returns non-linear advection term FEM intergral
x = zeros(ny,1);
x(1) = y(1)*y(2)+y(2)^2; 
for i = 2:ny-1
    x(i) = -y(i-1)^2-y(i-1)*y(i)+y(i)*y(i+1)+y(i+1)^2;
end
x(ny) = -y(ny-1)^2 - y(ny-1)*y(ny);
x = x/6;
end

