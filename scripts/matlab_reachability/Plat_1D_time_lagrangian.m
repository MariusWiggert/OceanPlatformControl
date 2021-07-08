%% Minimum-time backwards reachability in 1D system with uniform current
%
% Lagrangian = 1 and terminal cost =0 everywhere
%% Define the dynamical system
u_max = 1.;  % in m/s
x_target = 0;
fix_cur_magnitude = 0.;
dPlat = Plat_1D(x_target, u_max, fix_cur_magnitude);
%% Grid
grid_min = [-15];   % Lower corner of computation domain
grid_max = [15];    % Upper corner of computation domain
N = [40];         % Number of grid points per dimension
g = createGrid(grid_min, grid_max, N);

%% Different in this formulation
schemeData.hamFunc = @time_Ham;
data0 = zeros(g.shape);

%% time vector
t0 = 0;
tMax = 10;
dt = 1;
tau = t0:dt:tMax;
%% problem parameters
minWith = 'none'; % Backwards Reachable Set
% minWith = 'minVOverTime'; % Backwards Reachable Tube
%% backwards reachable set
uMode = 'min';
%% Pack problem parameters
% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = dPlat;
schemeData.accuracy = 'high'; %set accuracy
schemeData.uMode = uMode;
%% Compute value function
%HJIextraArgs.visualize = true; %show plot
% HJIextraArgs.visualize.valueSet = 1;
% HJIextraArgs.visualize.initialValueSet = 1;
% HJIextraArgs.visualize.figNum = 1; %set figure number
% HJIextraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update

%[data, tau, extraOuts] = ...
% HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)
[data, tau2, ~] = ...
  HJIPDE_solve(data0, tau, schemeData, minWith, HJIextraArgs);