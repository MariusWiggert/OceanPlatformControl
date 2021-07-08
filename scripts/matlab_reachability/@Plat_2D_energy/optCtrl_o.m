function uOpt = optCtrl(obj, ~, ~, deriv, uMode)
% uOpt = optCtrl(obj, t, y, deriv, uMode)

%% Input processing
if nargin < 5
  uMode = 'min';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

%% Optimal control
% => need to work with the index function!

if strcmp(uMode, 'max')
    error('Not implemented!')
elseif strcmp(uMode, 'min')
    [rows, cols] = size(deriv{obj.dims==1});
    A = [-1,1]';
    b = [1, 1]';
    uOpt = zeros(rows, cols);
    for i = 1:rows
        for j = 1:cols
            % set up function handle
            fun = @(u) get_Ham(obj, u, deriv{obj.dims==1}(i,j), deriv{obj.dims==2}(i,j));
            u_opt = fmincon(fun,[0],A,b);
            uOpt(i,j) = u_opt;
        end
    end
else
  error('Unknown uMode!')
end
end

function H=get_Ham(obj, u, px, pb)
         H = (obj.u_max*u + obj.x_current)*px + pb*(obj.c - obj.D .* abs(obj.u_max*u).^3) + obj.lambda*abs(obj.u_max*u).^3;
end