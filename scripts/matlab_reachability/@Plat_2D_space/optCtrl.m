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
% get the angle of the value function
theta = atan2(deriv{obj.dims==2},deriv{obj.dims==1});

if strcmp(uMode, 'max')
  error('Unknown uMode!')
elseif strcmp(uMode, 'min')
  uOpt = theta + pi;
else
  error('Unknown uMode!')
end

end