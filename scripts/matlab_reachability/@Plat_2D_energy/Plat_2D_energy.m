classdef Plat_2D_energy< DynSys
  properties
    % x is (x,b) position in x and battery level
    
    % Average battery charging per second 
    c
    
    % Drag coefficient D= 1/2*roh*c_d*A_ref
    D
   
    % maximum possible speed
    u_max
    
    % x_currents interpolant
    x_current
    
    % hyperparameter for running cost
    lambda
    
    % Dimensions that are active
    dims
  end
  
  methods
    function obj = Plat_2D_energy(x, u_max, fix_cur_magnitude, c, D, lambda, dims)
    % obj = SeaweedPlatform(x, u_max, fix_cur_magnitude, c, D)
    %     SeaweedPlatform class
    %
    % Dynamics of the Ocean Platform
    %    \dot{x}_1 = u_max*u + x_currents(x)
    %    \dot{x}_2 = I(b in [0,1])* (c - D * (u_max * u)^3)
    %         u in [-1, 1]
    %   Control: u = (u);
    %
    % Marius Wiggert, 2021-07-03
    % Inputs:
    %   x      - state: [xpos, battery_level]
    %   u_max - maximum speed in m/s
    %   fix_cur_magnitude - fixed current in x direction in m/s
    %   c - average charging of battery from solar per second
    %   D - Drag coefficient D= 1/2*roh*c_d*A_ref
    %
    % Output:
    %   obj       - a SeaweedPlatform object
      
      if numel(x) ~= obj.nx
        error('Initial state does not have right dimension!');
      end
      
      dims = 1:2;
      
      if ~iscolumn(x)
        x = x';
      end
      
      % Basic vehicle properties
      obj.pdim = 1; % Position dimensions
      obj.nx = length(dims);
      obj.nu = 1;  % will be 2 later
      obj.nd = 2;
      
      obj.c = c;
      obj.D = D;
      obj.lambda = lambda;
      
      obj.x_current = fix_cur_magnitude;
      
      obj.x = x;
      obj.xhist = obj.x;
      
      obj.u_max = u_max;
      obj.dims = dims;
    end
    
  end % end methods
end % end classdef
