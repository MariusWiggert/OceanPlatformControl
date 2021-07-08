%% Define the dynamical system
x_target_center = [-96.7, 22.5];
% xinit = [-96.3, 21.8]; % at the bottom right
xinit = [-96.8, 22.2]; % along the current
% xinit = [-96.9, 22.8];
file = 'gulf_of_mexico_2020-11-01-10_5h.nc4';
no_currents = 0; % true or false
u_max = 0.2;  % in m/s
dPlat = Plat_2D_space(x_target_center, u_max, file, no_currents);

% Stop when reached initial state
HJIextraArgs.stopInit = xinit;
%% Grid
grid_min = [dPlat.x_grid(1); dPlat.y_grid(1)]; % Lower corner of computation domain
grid_max = [dPlat.x_grid(end); dPlat.y_grid(end)];    % Upper corner of computation domain
N = [100; 100];         % Number of grid points per dimension
g = createGrid(grid_min, grid_max, N);

% %% Debugging: let's plot the 2D vector visualization
% x_matrix = dPlat.x_currents(g.xs{1, 1}, g.xs{2, 1});
% y_matrix = dPlat.y_currents(g.xs{1, 1}, g.xs{2, 1});
% quiver(g.xs{1, 1},g.xs{2, 1},x_matrix,y_matrix)
%% target set
R = 0.1;
% data0 = shapeCylinder(grid,ignoreDims,center,radius)
data0 = shapeCylinder(g, [], x_target_center, R);
% visSetIm(g, data0);
%% time vector (not it is in h this time)
t0 = 0;
tMax = 10;
dt = 1;
tau = t0:dt:tMax;
%% problem parameters
minWith = 'none'; % Backwards Reachable Set
% minWith = 'minVOverTime'; % Backwards Reachable Tube

% control trying to min or max value function?
uMode = 'min';
%% Pack problem parameters
% Put grid and dynamic systems into schemeData
schemeData.grid = g;
schemeData.dynSys = dPlat;
schemeData.accuracy = 'high'; %set accuracy
schemeData.uMode = uMode;
% if forward
% schemeData.tMode = 'forward';
% uMode = 'max';
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

%% Compute optimal trajectory from some initial state
% if compTraj

%check if this initial state is in the BRS/BRT
value = eval_u(g,data(:,:,end),xinit);
%%
if value <= 0 %if initial state is in BRS/BRT
    %find optimal trajectory

    dPlat.x = xinit; %set initial state of the dubins car

    TrajextraArgs.uMode = uMode; %set if control wants to min or max
    %     TrajextraArgs.dMode = 'max';
    TrajextraArgs.visualize = true; %show plot
    TrajextraArgs.fig_num = 2; %figure number

    %we want to see the first two dimensions (x and y)
    TrajextraArgs.projDim = [1 1]; 

    %flip data time points so we start from the beginning of time
    dataTraj = flip(data,3);

    % [traj, traj_tau] = ...
    % computeOptTraj(g, data, tau, dynSys, extraArgs)
    [traj, traj_tau] = ...
      computeOptTraj(g, dataTraj, tau2, dPlat, TrajextraArgs);

    %%
    figure(6)
    clf
    h = visSetIm(g, data(:,:,end));
    % h.FaceAlpha = .3;
    hold on
    s = scatter(xinit(1), xinit(2));
    s.SizeData = 70;
    title('The reachable set at the end and x_init')
    hold off

    %plot traj
    figure(4)
    plot(traj(1,:), traj(2,:))
    hold on
    xlim([grid_min(1) grid_max(1)])
    ylim([grid_min(2) grid_max(2)])
    % add the target set to that
    [g2D, data2D] = proj(g, data0, [0 0]);
    visSetIm(g2D, data2D, 'green');
    title('2D projection of the trajectory & target set')
    hold off
else
    error(['Initial state is not in the BRS/BRT! It have a value of ' num2str(value,2)])
end