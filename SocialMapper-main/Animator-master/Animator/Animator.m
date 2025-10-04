classdef (Abstract) Animator < FlexChart
    %Animator - Abstract superclass for data animation. Subclass of Chart.
    %
    %Animator Properties:
    %   frame - Frame of animation
    %   frameRate - Current frame rate.
    %   frameInds - Indices to use in frame s.t.
    %               currentFrame = obj.frameInds(obj.frame)
    %   scope - Identifier for current Animator in selective callbacks
    %   id - Integer 1-9, identifier for scope
    %   links - Cell array of linked Animators
    %   speedUp - Framerate increase rate.
    %   slowDown - Framerate decrease rate.
    %   ctrlSpeed - Framerate multiplier for ctrl.
    %   shiftSpeed - Framerate multiplier for shift.
    %
    %Animator methods:
    %   Animator - constructor
    %   delete - delete Animator
    %   get/set frame
    %   get/set frameRate
    %   restrict - restrict animation to subset of frames
    %   keyPressCallback - callback function
    %   writeVideo - Render a video of the current Animator and its links.
    %   update - Abstract method for updating frames in classes that
    %            inherit from Animator
    %   runAll - Static method for running the callback functions of all
    %            Animators in links
    %            It is useful to assign this function as the
    %            WindowKeyPressFcn of a figure with multiple Animators.
    %   linkAll - Link all Animators in a cell array together.
    properties (Access = protected)
        nFrames
    end
    
    properties (Access = public)
        frameInds
        frameRate = 1;
        frame = 1;
        id
        isVisible % UNUSED - REMOVE? vector representing visiblity of current frame
        % isVisible appears to never be updated - probably remove
        scope
        links
        speedUp = 10;
        slowDown = 10;
        ctrlSpeed = 50;
        shiftSpeed = 250;
    end
    
    methods
        function obj = Animator(varargin)
            %Animator - constructor for Animator abstract class.
            [flexChartArgs, ~, varargin] = parseClassArgs('FlexChart', varargin{:});
            obj@FlexChart(flexChartArgs{:});
            
            % User defined inputs
            if ~isempty(varargin)
                set(obj, varargin{:});
            end
            
            % Set up the figure and callback function
            obj.Parent = gcf;
            addToolbarExplorationButtons(gcf);
            set(obj.Parent, 'WindowKeyPressFcn', ...
                @(src, event) obj.keyPressCallback(src, event));
            
            % Set up the axes
            hold(obj.Axes, 'on');
            obj.Axes.DeleteFcn = @obj.onAxesDeleted;
            set(obj.Axes, 'Units', 'normalized');
        end
        
        function delete(obj)
            delete(obj.Axes);
        end % delete obj
        
        function axes = getAxes( obj )
            axes = obj.Axes;
        end
        
        % NOTE: you need to reimplement get.prop(obj) if you reimplement set.prop(obj)
        function frame = get.frame( obj )
            frame = obj.frame;
        end % get.frame
        
        function set.frame( obj, newFrame )
            % Set frame, but loop around before or after (adjust for 0 = nFrames)
            % also re-render the screen
            obj.frame = mod(newFrame, obj.nFrames);
            obj.frame(obj.frame == 0) = obj.nFrames;
            update(obj)
        end % set.frame
        
        function frameRate = get.frameRate( obj )
            frameRate = obj.frameRate;
        end % get.frame
        
        function set.frameRate( obj, newframeRate )
            %  update frame rate, but do not allow rate to go <1
            obj.frameRate = newframeRate;
            if obj.frameRate < 1
                obj.frameRate = 1;
            end
        end % set.frame
        
        function frameInds = getFrameInds(obj)
            frameInds = obj.frameInds;
        end
        
        function restrict(obj, newFrames)
            % change frameInds to a new vector (e.g. a subset range)
            % also update nFrames and reset frame number to 1
            obj.frameInds = newFrames;
            obj.nFrames = numel(newFrames);
            obj.frame = 1;
        end
        
        function keyPressCallback(obj, source, eventdata)
            % Determine the key that was pressed
            keyPressed = eventdata.Key;

            % The value updates are written this way to support
            % parallelization in very niche applications.
            % Consider rewriting.
            switch keyPressed
                case 'rightarrow'
                    obj.frame = obj.frame + obj.frameRate;
                case 'leftarrow'
                    obj.frame = obj.frame - obj.frameRate;
                case 'uparrow'
                    obj.frameRate = obj.frameRate + obj.speedUp;
                case 'downarrow'
                    obj.frameRate = obj.frameRate - obj.speedUp;
                case {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
                    val = str2double(keyPressed);
                    obj.scope = val;
                    fprintf('Scope is Animation %d\n', val);
            end
            set(obj.Axes.Parent , 'NumberTitle', 'off', ...
                'Name', sprintf('Frame: %d', obj.frameInds(obj.frame(1))));
        end
        
        function checkVisible(obj)
            % check if the current frame is visible in the isVisible vector
            % and update the current axes & axes.children to that value
            % Typically run inside subclass's `obj.update` method (i.e. every rerender)

            % NOTE: this is probably NEVER RUN since isVisible is never changed from the uninitialized value: []
            if ~isempty(obj.isVisible)
                vis = obj.isVisible(obj.frameInds(obj.frame));
                arrayfun(@(X) set(X, 'Visible', vis), obj.Axes.Children)
                set(obj.Axes, 'Visible', vis)
            end
        end
    end
    
    methods
        % video-related methods: not used in Label3D
        function V = writeVideo(obj, frameIds, savePath, varargin)
            %writeMovie - write an Animator movie
            %
            %   Syntax: Animator.writeMovie(frameIds, savePath, 'FPS', 30);
            %
            %   Inputs: frameIds - frames to write
            %           savePath - path to save movie
            %           varargin - arguments to write_frames.m.
            %
            %   Notes: The writing function for .avi is different than the
            %   one for .gifs. Be sure to use the correct varargs E.g.
            %   "DelayTime" for gifs, "FPS" for avi.
            %
            %   Required .m files: write_frames.m
            
            % Find all of the linked Animators
            linkedAnimators = obj.links;
            if (numel(linkedAnimators) == 1) && (~iscell(linkedAnimators))
                linkedAnimators = {linkedAnimators};
            end
            if isempty(linkedAnimators)
                linkedAnimators = cell(1);
                linkedAnimators{1} = obj;
            end
            V = cell(numel(frameIds), 1);
            tic
            % Iterate through frames, update each Animator, and get the
            % image.
            origFrame = linkedAnimators{1}.frame;
            for nFrame = 1 : numel(frameIds)
                % Sometimes you might want to change this syntax if you've
                % restricted frames via Animator.restrict.
                for nAnimator = 1 : numel(linkedAnimators)
                    linkedAnimators{nAnimator}.frame = frameIds(nFrame);
%                     linkedAnimators{nAnimator}.frame = linkedAnimators{nAnimator}.frame + 1;
                end
%                 for nAnimator = 1:numel(linkedAnimators)
% %                     linkedAnimators{nAnimator}.frame = linkedAnimators{nAnimator}.frame+1;
%                     linkedAnimators{nAnimator}.frame = origFrame+(frameIds(nFrame));
%                 end
                
                % Grab the image
                F = getframe(obj.Parent);
                V{nFrame} = F.cdata;
                
                % Print out an estimate of rendering time.
                if nFrame == 100
                    rate = 100 / (toc);
                    fprintf('Estimated time remaining: %f seconds\n', ...
                        numel(frameIds)/rate)
                end
            end
            % Aggregate the movie and write.
            V = cat(4, V{:});
            fprintf('Writing movie to: %s\n', savePath);
            if strcmp(get_ext(savePath), '.gif')
                for nFrame = 1 : size(V, 4)
                    im = squeeze(V(:, :, :, nFrame));
                    [imind, cm] = rgb2ind(im, 256);
                    if nFrame == 1
                        imwrite(imind, cm, savePath, 'gif', 'Loopcount', inf, varargin{:});
                    else
                        imwrite(imind, cm, savePath, 'gif', 'WriteMode', 'append', varargin{:});
                    end
                end
            else
                write_frames(V, savePath, varargin{:}); 
            end
        end
        
        function play(obj, frameIds)
            %play - play an Animator movie
            %
            %   Syntax: Animator.play(frameIds);
            %
            %   Inputs: frameIds - frames to play
            
            % Find all of the linked Animators
            linkedAnimators = obj.links;
            if (numel(linkedAnimators) == 1) && (~iscell(linkedAnimators))
                linkedAnimators = {linkedAnimators};
            end
            if isempty(linkedAnimators)
                linkedAnimators = cell(1);
                linkedAnimators{1} = obj;
            end
            tic
            % Iterate through frames, update each Animator
            origFrame = linkedAnimators{1}.frame;
            for nFrame = 1 : numel(frameIds)
                % Sometimes you might want to change this syntax if you've
                % restricted frames via Animator.restrict.
                for nAnimator = 1 : numel(linkedAnimators)
                    linkedAnimators{nAnimator}.frame = frameIds(nFrame);
                    
                end
                drawnow;
            end
        end
    end
    
    methods (Abstract, Access = protected)
        update(obj)
    end
    
    methods (Static)
        function linkAll(animatorList)
            % update the links value of all animators
            % and set each animator's parent (E.g. "Label3d GUI" to runAll animator keyPressCallback's)
            for i = 1:numel(animatorList)
                animatorList{i}.links = animatorList;
            end
            cellfun( ...
                @(X) set( ...
                    X.Parent, ...
                    'WindowKeyPressFcn', ...
                    @(src, event) Animator.runAll(animatorList, src, event) ...
                ), animatorList)
        end
        
        % NOTE: never used in Label3D
        function tileAnimators(animatorList, varargin)
            % position all animators specific by an cell array of animator handles 
            % varargin = {pad}
            nAnimators = numel(animatorList);
            pad = 0.05;
            if ~isempty(varargin)
                pad = varargin{1};
            end
            w = 1 / nAnimators - 2 * pad;
            starts = (1 / nAnimators) * (0 : (nAnimators - 1)) + pad;
            for nAnimator = 1 : nAnimators
                ax = animatorList{nAnimator}.getAxes();
                % set position syntax: [left, bottom, width, height]
                set(ax, 'Position', [starts(nAnimator), pad/2, w, 1-2*pad])
            end
        end
        
        function runAll(animatorList, src, event)
            %runAll - iterate through the keyPressCallback function of all
            %Animators within a cell array.
            %
            %   Syntax: runAll(animatorList, src, event);
            %
            %   Notes: It is useful to assign this function as the
            %          WindowKeyPressFcn of a figure with multiple axes
            %          that listen for key presses.
            for i = 1 : numel(animatorList)
                if isa(animatorList{i}, 'Animator')
                    animatorList{i}.keyPressCallback(src, event)
                end
            end
        end
    end
end