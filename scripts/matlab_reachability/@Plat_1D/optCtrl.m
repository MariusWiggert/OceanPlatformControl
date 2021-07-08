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
% x = magic(3)-5;
% idx = sign(x);
% x(idx==1) = x(idx==1).^2;
% x(idx==-1) = x(idx==-1).^3;

% T = M1 - M2;
% S = zeros(numel(M1),1);
% l = abs(T) >= Theta;
% S(l) = sign(T(l));

% => need to work with the index function!

if strcmp(uMode, 'max')
    uOpt = - (deriv{obj.dims==1} < 0) +  (deriv{obj.dims==1} > 0);
elseif strcmp(uMode, 'min')
    uOpt = + (deriv{obj.dims==1} < 0) -  (deriv{obj.dims==1} > 0);
else
  error('Unknown uMode!')
end

end