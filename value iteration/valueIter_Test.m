%% Initialize the MDP

% Transition matrix
P = [0.2 0.8 0 0; 0 0 0 0; 0 0 0 0; 0 0 0.9 0.1];
P(:,:,2) = [0.2 0 0 0.8; 0 0.2 0.8 0; 0 0 0 0; 0 0 0 0];
P(:,:,3) = [0 0 0 0; 0.8 0.2 0 0; 0 0 0 1; 0 0 0 0];
P(:,:,4) = [0 0 0 0; 0 0 0 0; 0 1 0 0; 0.8 0 0 0.2];

% Reward matrix
R = [0 0 0 0; 0 0 0 0; 0 0 1 1; 0 0 0 0];

S = size(P,1);
A = size(P,3);

% Discount factor
discount = 0.5;

% Initialize vector of initial state values
V0 = 10 * rand(S,1) -5;

% Epsilon
epsilon = 1e-6;

% Maxiteration
maxiterations = 100;

%% Execute value iteration algorithm
[vSync, VHSync, counterSync, errorSync] = valueIter(P, R, discount, epsilon, maxiterations, V0);

[vPlace, VHPlace, counteP, errorP] = valueIter(P, R, discount, epsilon, maxiterations, V0, 'InPlace');

