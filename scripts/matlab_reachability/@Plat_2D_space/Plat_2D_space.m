classdef Plat_2D_space < DynSys
  properties
    % x is (x,y) position in in deg longitude, latitude
   
    % maximum possible speed
    u_max
    
    % x_currents interpolant
    x_currents
    
    % y_currents interpolant
    y_currents
    
    % grid stuff
    x_grid
    y_grid
    t_grid
    
    % static stuff
    conv_m_to_deg = 111120.
    
    % Dimensions that are active
    dims
  end
  
  methods
    function obj = Plat_2D_space(x, u_max, current_file, no_currents)
      % obj = SeaweedPlatform(x, u_max, current_file)
      %     SeaweedPlatform class
      %
      % Dynamics:
      %    \dot{x}_1 = u_max*cos(alpha) + x_currents(x,y)
      %    \dot{x}_2 = u_max*sin(alpha) + y_currents(x,y)
      %         u_max is fixed for now, u < u_max => will be implemented later
      %         alpha in [0, 2pi]
      %
      % Inputs:
      %   x      - state: [xpos; ypos]
      %   u_max - maximum speed (for starters constant)
      %   current_file - path to nc4 file for currents
      %
      % Output:
      %   obj       - a SeaweedPlatform object
      
      if numel(x) ~= obj.nx
        error('Initial state does not have right dimension!');
      end

      dims = 1:2;
      
      if nargin < 4
        no_currents = 0;
      end
      
      
      if ~iscolumn(x)
        x = x';
      end
      
      % Basic vehicle properties
      obj.pdim = [1 2]; % Position dimensions
      obj.nx = length(dims);
      obj.nu = 1;   % will be 2 later
      obj.nd = 2;
      
      % create the current interpolation functions
      obj.x_grid = ncread(current_file,'lon');
      obj.y_grid = ncread(current_file,'lat');
      obj.t_grid = ncread(current_file,'time');
      if ~no_currents
          water_u = squeeze(ncread(current_file,'water_u'));
          water_u_static = water_u(:,:,1);
          water_u_static(isnan(water_u_static))=0;
          water_v = squeeze(ncread(current_file,'water_v'));
          water_v_static = water_v(:,:,1);
          water_v_static(isnan(water_v_static))=0;
      else
          water_u_static = zeros(length(obj.x_grid), length(obj.y_grid));
          water_v_static = zeros(length(obj.x_grid), length(obj.y_grid));
      end
       
      % TODO: lateron change to (x,y,t)
      % Note: this is in m/s
%       obj.x_currents = @(x,y) interpn(obj.x_grid,obj.y_grid,obj.t_grid,water_v, x, y, 182652,'linear');
%       obj.y_currents = @(x,y) interpn(obj.x_grid,obj.y_grid,obj.t_grid,water_u, x, y, 182652,'linear');
      obj.x_currents = @(x,y) interpn(obj.x_grid,obj.y_grid,water_u_static, x, y,'linear');
      obj.y_currents = @(x,y) interpn(obj.x_grid,obj.y_grid,water_v_static, x, y,'linear');
      
      obj.x = x;
      obj.xhist = obj.x;
      
      obj.u_max = u_max;
      obj.dims = dims;
    end
    
  end % end methods
end % end classdef
