nsample = 3;
 num_output = 96; % conv1
% num_output = 256; % conv5
%num_output = 4096; % fc7

load features_conv1.mat
width = size(feats, 2);
nmap = width / num_output;

for i = 1 : nsample
    feat = feats(i, :);
    feat = reshape(feat, [nmap num_output]);
    figure('name', sprintf('image #%d', i));
    display_network(feat);
end