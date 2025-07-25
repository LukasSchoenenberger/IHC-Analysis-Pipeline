function move_dcm_files_to_subfolder(motherFolder)
    % Check if the input is a valid directory
    if ~isfolder(motherFolder)
        error('The specified motherFolder is not a valid directory.');
    end

    % Get all subfolders in the given motherFolder
    subfolders = dir(fullfile(motherFolder, '**', '*'));
    subfolders = subfolders([subfolders.isdir]);  % Keep only directories

    % Loop through each subfolder
    for i = 1:length(subfolders)
        folderPath = fullfile(subfolders(i).folder, subfolders(i).name);
        
        % Skip "." and ".." folders
        if strcmp(subfolders(i).name, '.') || strcmp(subfolders(i).name, '..')
            continue;
        end

        % Check if the "DCM" subfolder already exists; if not, create it
        dcmFolderPath = fullfile(folderPath, 'DCM');
        if ~isfolder(dcmFolderPath)
            mkdir(dcmFolderPath);
            fprintf('Created DCM folder in: %s\n', folderPath);
        end

        % Get all .dcm files in the current subfolder
        dcmFiles = dir(fullfile(folderPath, '*.dcm'));

        % Move each .dcm file to the "DCM" subfolder
        for j = 1:length(dcmFiles)
            dcmFilePath = fullfile(folderPath, dcmFiles(j).name);
            newDcmFilePath = fullfile(dcmFolderPath, dcmFiles(j).name);

            % Check if the file is already in the DCM folder (skip if it is)
            if ~strcmp(dcmFilePath, newDcmFilePath)
                movefile(dcmFilePath, newDcmFilePath);
                fprintf('Moved DCM file: %s -> %s\n', dcmFilePath, newDcmFilePath);
            end
        end
    end
end
