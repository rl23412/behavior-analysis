function [xx,density,G,Z] = findPointDensity(points,sigma,numPoints,rangeVals)

    % Handle grid size input: allow scalar or [nx ny]
    if nargin < 3 || isempty(numPoints)
        nx = 1001; ny = 1001;
    else
        if numel(numPoints) == 1
            nx = numPoints; ny = numPoints;
        else
            nx = numPoints(1); ny = numPoints(2);
        end
        if mod(nx,2) == 0; nx = nx + 1; end
        if mod(ny,2) == 0; ny = ny + 1; end
    end
    
    % Handle range input: allow [-a a] or [xmin xmax ymin ymax]
    if nargin < 4 || isempty(rangeVals)
        rangeVals = [-110 110];
    end
    if numel(rangeVals) == 2
        x = linspace(rangeVals(1),rangeVals(2),nx);
        y = x;
    else
        x = linspace(rangeVals(1),rangeVals(2),nx);
        y = linspace(rangeVals(3),rangeVals(4),ny);
    end

    % Build grids
    [XX,YY] = meshgrid(x,y);
    
    % Gaussian kernel
    G = exp(-.5.*(XX.^2 + YY.^2)./sigma^2) ./ (2*pi*sigma^2);
    
    % 2D histogram of points
    Z = hist3(points,{x,y});
    Z = Z ./ (sum(Z(:)));
    
    % Convolve and normalize
    density = fftshift(real(ifft2(fft2(G).*fft2(Z))))';
    density(density<0) = 0;
    
    % Return the 1D coordinate vector for convenience
    xx = x;
    
    %imagesc(x,y,density)
    %axis equal tight
    %set(gca,'ydir','normal');