function dx = dynamics(obj, ~, x, u, d)
% Dynamics of the Ocean Platform
%    \dot{x}_1 = u_max*cos(alpha) + x_currents(x,y)
%    \dot{x}_2 = u_max*sin(alpha) + y_currents(x,y)
%         u_max is fixed for now, u < u_max => will be implemented later
%         alpha in [0, 2pi]
%   Control: u = (alpha)';
%
% Marius Wiggert, 2021-06-21

if nargin < 5
  d = [0; 0];
end

if iscell(x)
  dx = cell(length(obj.dims), 1);
  
  for i = 1:length(obj.dims)
    dx{i} = dynamics_cell_helper(obj, x, u, d, obj.dims, obj.dims(i));
  end
else
  dx = zeros(obj.nx, 1);
  
  dx(1) = (obj.u_max * cos(u(1)) + obj.x_currents(x(1), x(2)))*(3600/obj.conv_m_to_deg);
  dx(2) = (obj.u_max * sin(u(1)) + obj.y_currents(x(1), x(2)))*(3600/obj.conv_m_to_deg);

% later when u is variable
%   dx(1) = u(1)*obj.u_max * cos(u(2)) + d(1);
%   dx(2) = u(2)*obj.u_max * sin(u(2)) + d(2);
end
end

function dx = dynamics_cell_helper(obj, x, u, d, dims, dim)

switch dim
  case 1
    dx = (obj.u_max * cos(u) + obj.x_currents(x{1}, x{2}))*(3600/obj.conv_m_to_deg);
  case 2
    dx = (obj.u_max * sin(u) + obj.y_currents(x{1}, x{2}))*(3600/obj.conv_m_to_deg);
  otherwise
    error('Only dimension 1-2 are defined for dynamics of Ocean Platform!')
end
end