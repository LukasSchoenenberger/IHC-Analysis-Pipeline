function run_dcm2niix_on_subfolders(motherFolder)
    % Check if the input is a valid directory
    if ~isfolder(motherFolder)
        error('The specified motherFolder is not a valid directory.');
    end

    % Get all subfolders in the given motherFolder
    subfolders = dir(fullfile(motherFolder, '**', '*'));
    subfolders = subfolders([subfolders.isdir]);  % Keep only directories

    % For each subfolder
    for i = 1:length(subfolders)
        folderPath = fullfile(subfolders(i).folder, subfolders(i).name);
        
        % Skip "." and ".." folders
        if strcmp(subfolders(i).name, '.') || strcmp(subfolders(i).name, '..')
            continue;
        end

        % Run the dcm2niix system command on the subfolder
        fprintf('Running dcm2niix on: %s\n', folderPath);
        
        % Construct the system command
        command = sprintf('dcm2niix "%s"', folderPath);
        
        % Execute the system command
        [status, cmdout] = system(command);
        
        % Check if the command was successful
        if status == 0
            fprintf('dcm2niix finished successfully for: %s\n', folderPath);
        else
            fprintf('Error running dcm2niix on %s: %s\n', folderPath, cmdout);
        end
    end
end
