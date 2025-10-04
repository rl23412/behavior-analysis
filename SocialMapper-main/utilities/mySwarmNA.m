function [outinfo, tt] = mySwarmNA(C, cents, jitterWidth, colors)  
    % mySwarmNA - Create swarm plots for behavioral data visualization  
    %   
    % Inputs:  
    %   C - cell array where each cell contains data for one group  
    %   cents - x-axis center positions for each group  
    %   jitterWidth - width of jitter for swarm effect  
    %   colors - color matrix (nGroups x 3) for each group  
    %  
    % Outputs:  
    %   outinfo - structure with plot information  
    %   tt - handles to plotted objects  
      
    if nargin < 4  
        colors = lines(length(C));  
    end  
      
    if nargin < 3  
        jitterWidth = 0.3;  
    end  
      
    nGroups = length(C);  
    tt = cell(nGroups, 1);  
    outinfo = struct();  
      
    hold on;  
      
    for i = 1:nGroups  
        data = C{i};  
        if isempty(data)  
            continue;  
        end  
          
        % Remove NaN values  
        data = data(~isnan(data));  
        n = length(data);  
          
        if n == 0  
            continue;  
        end  
          
        % Create jitter for x-positions  
        if n == 1  
            xJitter = 0;  
        else  
            % Use a simple algorithm to spread points  
            xJitter = linspace(-jitterWidth/2, jitterWidth/2, n);  
            % Add some randomness to avoid perfect alignment  
            xJitter = xJitter + (rand(1,n) - 0.5) * jitterWidth * 0.1;  
        end  
          
        % X positions centered at cents(i)  
        xPos = cents(i) + xJitter;  
          
        % Plot the swarm  
        if size(colors, 1) >= i  
            color = colors(i, :);  
        else  
            color = [0.5 0.5 0.5]; % default gray  
        end  
          
        % Create scatter plot  
        h = scatter(xPos, data, 30, color, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);  
        tt{i} = h;  
          
        % Store statistics  
        outinfo.means(i) = mean(data);  
        outinfo.medians(i) = median(data);  
        outinfo.stds(i) = std(data);  
        outinfo.ns(i) = n;  
          
        % Optionally add mean line  
        line([cents(i) - jitterWidth/3, cents(i) + jitterWidth/3], ...  
             [outinfo.means(i), outinfo.means(i)], ...  
             'Color', color * 0.7, 'LineWidth', 2);  
    end  
      
    % Set axis properties  
    if nGroups > 0  
        xlim([min(cents) - 1, max(cents) + 1]);  
        set(gca, 'XTick', cents);  
    end  
      
    % Store additional info  
    outinfo.centers = cents;  
    outinfo.colors = colors;  
    outinfo.nGroups = nGroups;  
end