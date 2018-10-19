function [values, valueHistory, counter, error] = valueIter(P, R, discount, epsilon, maxiterations, V0, updateType)
% Compute the state-value using value iteration algorithm. Two version of
% the algorithm are available: Synchronized update or In place update.
% The latter use only one vector of state-values to compute the updates.

% Arguments --------------------------------------------------------------
% Let S = number of states, A = number of actions
%   P(SxSxA) = transition matrix 
%   R(SxA) = reward matrix          
%   discount  = discount rate in [0;1]
%
%   epsilon (OPT) = max ( |V - V*| ) < epsilon,  upper than 0 (default : 0.01)
%   maxiterations (OPT) = maximum number of iterations allowed (default : 1000)
%   V0(S)(OPT) = starting value function (default : zeros(S,1))
%
% Output ------------------------------------------------------------------
%
%   values = state-values 
%   valueHistory = list of all values assigned to states during the process
%   counter = total number of iteration performed.
%   error = the error that is still present in the last iteration 


% Set param S and A to be respectively the number of states and actions
S = size(P,1);
A = size(P,3);

% check of arguments
if discount < 0 || discount > 1
    disp('--------------------------------------------------------')
    disp('Discount rate must be in [0;1]')
    disp('--------------------------------------------------------')
elseif nargin > 3 && (epsilon < 0)
    disp('--------------------------------------------------------')
    disp('epsilon must be upper than 0')
    disp('--------------------------------------------------------')
elseif nargin > 6 && maxiterations < 0
    disp('--------------------------------------------------------')
    disp('The number of max iteration must be greater than 0')
    disp('--------------------------------------------------------')
elseif nargin > 5 && size(V0,1) ~= S
    disp('--------------------------------------------------------')
    disp('V0 must have the same dimension as P')
    disp('--------------------------------------------------------') 
elseif nargin > 6 && not ( strcmp(updateType,'InPlace') )
    disp('--------------------------------------------------------')
    disp('Only InPlace is allowed as alternative update type')
    disp('--------------------------------------------------------') 
end

  % set default values
  if nargin < 7; updateType = 'Sync'; end
  if nargin < 6; V0 = zeros(S,1); end
  if nargin < 5; maxiterations = 1000; end
  if nargin < 4; epsilon = 0.01; end
  
  %Preallocate valueHistory and initialize it to contains the first values.
  valueHistory = zeros(maxiterations,S);
  valueHistory(1,:) = V0';
  
  % Vector holding final estimation of the values of the states.
  values = zeros(S,1);
  
  % Initialize counter and set error to a very high number.
  counter = 0;
  error = 1e300;
  
  
  
  %% IN PLACE VALUE ITERATION
  if strcmp(updateType,'InPlace')
      
    while error > epsilon && counter < maxiterations
        
        counter = counter + 1;
        error = 0;
        
        for s = 1:S
            % Select section of the 3d matrix which contains trasactions involving state s
            PTemp = reshape( P(s,:,:), S, A); 

            % Select section of 2d matrix R which contains rewards for action starting from state s s 
            RTemp = R(s,:); 
            
            % Temporarily store the last value of the state to calculate the error.
            last = values(s);
            
            % Calculate the new value for the state s
            values(s) = max( RTemp' + discount * (PTemp' * values) );
            
            % Store the max improvement in State-value obtain during the last iteration
            lastUpdate = abs ( values(s) - last );
            if ( lastUpdate > error ) 
                error = lastUpdate;
            end
            
        end
        % Hang the updated estimate of the values to the history
        valueHistory(counter + 1,:) = values';
    end
     
      
     
    
  %% SYNCHRONIZED VALUE ITERATION 
  else
      
      while error > epsilon && counter < maxiterations

        counter = counter + 1;

        for s = 1:S
            % Select section of the 3d matrix which contains trasactions involving state s
            PTemp = reshape( P(s,:,:), S, A); 

            % Select section of 2d matrix R which contains rewards for action starting from state s s 
            RTemp = R(s,:); 

            % Calculate the new value for the state s
            values(s) = max( RTemp' + discount * (PTemp' * V0) ); 
        end

        % Update the error
        error = max( abs(V0 - values));

        % Update the estimate of the state's value to be used in the next iter
        V0 = values;

        % Hang the updated estimate of the values to the history
        valueHistory(counter + 1,:) = values';

      end
  
  end
  
  %Erase preallocated cell of valueHistory not used
  valueHistory = valueHistory(1:counter,:); 
  
end %of function
