function dx = dynamics(obj, ~, x, u, d)
% Dynamics of the Ocean Platform
%    \dot{x}_1 = u_max*u + x_currents(x)
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
end
end

function dx = dynamics_cell_helper(obj, x, u, d, dims, dim)

switch dim
  case 1
    dx = obj.u_max * u + obj.x_current;
  otherwise
    error('Only dimension 1 is defined for dynamics of Ocean Platform_1D!')
end
end