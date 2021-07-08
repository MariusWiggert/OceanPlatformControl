%% Define the dynamical system
u_max = 1.;     % in m/s
c = 0.;         % charging of battery per second
D = 1.;         % Drag coefficient D= 1/2*roh*c_d*A_ref
lambda = 0.;    % Additional Battery running cost L= lambda*u^3 (or b?)
schemeData.hamFunc = @energy_Ham;
x_init = [0,5];
x_target_center = [0,5];
fix_cur_magnitude = 0.;
dPlat = Plat_2D_energy(x_init, u_max, fix_cur_magnitude, c, D, lambda);
%% Grid
grid_min = [-10; -1]; % Lower corner of computation domain
grid_max = [10; 11];    % Upper corner of computation domain
N = [50; 50];        % Number of grid points per dimension
g = createGrid(grid_min, grid_max, N);
%% target set
R = .5;
% data0 = shapeRectangleByCenter(g, x_target_center, R);
data0 = shapeCylinder(g, [], x_target_center, R);
% visSetIm(g, data0);

%% add obstacles
normal_upper = [0, -1]';
point_upper = [0, 10]';
upper = shapeHyperplane(g, normal_upper, point_upper);

normal_lower = [0, 1]';
point_lower = [0, 0]';
lower = shapeHyperplane(g, normal_lower, point_lower);

% merge them with a max
obstacles = min(upper, lower);
% visSetIm(g, obstacles);
% visFuncIm(g,-obstacles, 'red', .5)
% hold on
% visFuncIm(g,data0, 'blue', .5)
HJIextraArgs.obstacles = obstacles;
% HJIextraArgs.obstacle_mask = 1;
%% time vector
t0 = 0;
tMax = 5;
dt = 0.5;
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
% HJIextraArgs.visualize.valueFunction = 1;
HJIextraArgs.visualize.figNum = 1; %set figure number
HJIextraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update
HJIextraArgs.visualize.xTitle = 'x in distance';
HJIextraArgs.visualize.yTitle = 'Battery level';

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