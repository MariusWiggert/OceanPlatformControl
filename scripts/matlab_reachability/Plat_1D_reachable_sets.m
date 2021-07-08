% We do reachability in a 1D system with a uniform current
%
% Backwards reachability: we set the as terminal cost a signed distance 
%                         function from the target. When running we do
%                         uMode = 'min' as we want to be inside the target
%                         also set the schemeData.tMode = 'forward';
%
% Forward reachability:   we set the as initial cost/constraint a 
%                         signed distance function from the start point. 
%                         When running we do uMode = 'max'. This is solved
%                         as a backwards-backwards problem with the initial 
%                         as terminal cost and the goal being to max how
%                         far away the vehicle can get from the start at
%                         the end of time. 
%                         Comment out schemeData.tMode = 'forward';
%% Define the dynamical system
u_max = 1.;  % in m/s
x_init = 5;
fix_cur_magnitude = 1.;
dPlat = Plat_1D(x_init, u_max, fix_cur_magnitude);
%% Grid
grid_min = [-15];   % Lower corner of computation domain
grid_max = [15];    % Upper corner of computation domain
N = [40];         % Number of grid points per dimension
g = createGrid(grid_min, grid_max, N);
%% target set
R = 1;
% data0 = shapeCylinder(grid,ignoreDims,center,radius)
data0 = shapeCylinder(g, [], [0], R);
% visSetIm(g, data0);
%% time vector
t0 = 0;
tMax = 5;
dt = 0.5;
tau = t0:dt:tMax;
%% problem parameters
minWith = 'none'; % Backwards Reachable Set
% minWith = 'minVOverTime'; % Backwards Reachable Tube

% control trying to min or max value function?
%% backwards reachable set
uMode = 'min';
%% when doing forward reachable set
% uMode = 'max'; 
% schemeData.tMode = 'forward';
%% Pack problem parameters
% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = dPlat;
schemeData.accuracy = 'high'; %set accuracy
schemeData.uMode = uMode;
%% Compute value function
%HJIextraArgs.visualize = true; %show plot
HJIextraArgs.visualize.valueSet = 1;
HJIextraArgs.visualize.initialValueSet = 1;
HJIextraArgs.visualize.figNum = 1; %set figure number
HJIextraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update

%[data, tau, extraOuts] = ...
% HJIPDE_solve(data0, tau, schemeData, minWith, extraArgs)
[data, tau2, ~] = ...
  HJIPDE_solve(data0, tau, schemeData, minWith, HJIextraArgs);