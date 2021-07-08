function dx = dynamics(obj, ~, x, u, d)
% Dynamics of the Ocean Platform
%    \dot{x}_1 = u_max*u + x_currents(x)
%    \dot{x}_2 = I(b in [0,1])* (c - D * (u_max * u)^3)
%         u in [-1, 1]
%   Control: u = (u);
%
% Marius Wiggert, 2021-07-03

if nargin < 5
  d = [0];
end

if iscell(x)
  dx = cell(length(obj.dims), 1);
  
  for i = 1:length(obj.dims)
    dx{i} = dynamics_cell_helper(obj, x, u, d, obj.dims, obj.dims(i));
  end
else
  dx = zeros(obj.nx, 1);
  
  dx(1) = obj.u_max * u(1) + obj.x_current;
  % check if that works
  % might lead to issues of getting stuck below 0 and above 1
  % dx(2) = (x(2) >= 0 && x(2) <=1) * (obj.c - obj.D * (obj.u_max*u(1))^3);
  dx(2) = (obj.c - obj.D * abs(obj.u_max*u(1)).^3);
  
end
end

function dx = dynamics_cell_helper(obj, x, u, d, dims, dim)

switch dim
  case 1
    dx = obj.u_max .* u + obj.x_current;
  case 2
    dx = (obj.c - obj.D .* abs(obj.u_max*u).^3);
%     dx = (x{2} >= 0 & x{2} <=1) .* (obj.c - obj.D .* abs(obj.u_max*u).^3);
  otherwise
    error('Only 2 dimensions are defined!')
end
end