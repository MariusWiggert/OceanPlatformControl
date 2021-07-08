classdef Plat_1D< DynSys
  properties
    % x is (x,y) position in in deg longitude, latitude
   
    % maximum possible speed
    u_max
    
    % x_currents interpolant
    x_current
    
    % Dimensions that are active
    dims
  end
  
  methods
    function obj = Plat_1D(x, u_max, fix_cur_magnitude, dims)
    % obj = SeaweedPlatform(x, u_max, current_file)
    %     SeaweedPlatform class
    %
    % Dynamics of the Ocean Platform
    %    \dot{x}_1 = u + x_currents(x)
    %         u in [-1, 1]
    %   Control: u = (u);
    % Inputs:
    %   x      - state: [xpos]
    %   u_max - maximum speed in m/s
    %   fix_cur_magnitude - fixed current in x direction in m/s
    %
    % Output:
    %   obj       - a SeaweedPlatform object
      
      if numel(x) ~= obj.nx
        error('Initial state does not have right dimension!');
      end
      
      if nargin < 4
        dims = [1];
      end
      
      if ~iscolumn(x)
        x = x';
      end
      
      % Basic vehicle properties
      obj.pdim = [1]; % Position dimensions
      obj.nx = length(dims);
      obj.nu = 1;   % will be 2 later
      obj.nd = 1;
      
      obj.x_current = fix_cur_magnitude;
      
      obj.x = x;
      obj.xhist = obj.x;
      
      obj.u_max = u_max;
      obj.dims = dims;
    end
    
  end % end methods
end % end classdef
