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
    % u_mid_non_abs for the both derivatives negative case
    u_mid = (abs(deriv{obj.dims==1})./(3.*abs(deriv{obj.dims==2}.*obj.D - obj.lambda))).^0.5;
    u_mid = real(u_mid);
    u_mid(isinf(u_mid) | isnan(u_mid)) = 0;
    
    both_pos = deriv{obj.dims==1} > 0 & (deriv{obj.dims==2}*obj.D-obj.lambda) > 0;
    both_neg = deriv{obj.dims==1} < 0 & (deriv{obj.dims==2}*obj.D-obj.lambda) < 0;
    pos_neg = deriv{obj.dims==1} > 0 & (deriv{obj.dims==2}*obj.D-obj.lambda) < 0;
    neg_pos = deriv{obj.dims==1} < 0 & (deriv{obj.dims==2}*obj.D-obj.lambda) > 0;
    
    % Extreme values can be -1., -u_mid, 0, +u_mid, +1
    H_min_1 = H(obj, -1., deriv);
    H_mid_neg = H(obj, -u_mid.*(u_mid <=1), deriv);
    H_mid_pos = H(obj, u_mid.*(u_mid <=1), deriv);
    H_plus_1 = H(obj, 1., deriv);
    
    uOpt = zeros(size(deriv{obj.dims==1}));
    uOpt = uOpt + both_pos.*((-1).*(H_min_1 < H_plus_1) + (1).*(H_min_1 > H_plus_1));
    uOpt = uOpt + both_neg.*(u_mid.*(u_mid <=1).*(H_mid_pos < H_plus_1) + (1).*(H_mid_pos > H_plus_1));
    uOpt = uOpt + pos_neg.*((-1).*(H_min_1 < H_mid_neg) + -u_mid.*(u_mid <=1).*(H_min_1 > H_mid_neg));
    uOpt = uOpt + neg_pos.*((-1).*(H_min_1 < H_plus_1) + (1).*(H_min_1 > H_plus_1));
    
%     
%     % now compare for each of the values
%     all_concat = cat(3,H_min_1,H_mid_abs,H_0,H_mid_pos,H_plus_1);
%     min_values = min(all_concat,[],3);
%     
%     % sort them in with index functions (can also do all together)
%     uOpt = zeros(size(deriv{obj.dims==1}));
%     uOpt = uOpt + (-1).*(min_values == H_min_1);
%     uOpt = uOpt + (0).*(min_values == H_0);
%     uOpt = uOpt + (u_mid_pos).*(min_values == H_mid_pos & (u_mid_pos <=1));
%     uOpt = uOpt + (-u_mid_abs).*(min_values == H_mid_abs & (u_mid_abs <=1));
%     uOpt = uOpt + (1).*(min_values == H_plus_1);

else
  error('Unknown uMode!')
end
% if length(uOpt)>2
%     figure
%     heatmap(uOpt)
%     keyboard
% end
end

function H=H(obj, u, deriv)
    dx1 = obj.u_max .* u + obj.x_current;
    dx2 = (obj.c - obj.D .* abs(obj.u_max*u).^3);
    H = deriv{1}.*dx1 + deriv{2}.*dx2 + obj.lambda.*abs(obj.u_max.*u).^3;
end
