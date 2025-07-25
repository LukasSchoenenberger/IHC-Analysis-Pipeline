function rename_subfolders_based_on_json(motherFolder)
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

        % Look for the first .json file in the subfolder
        jsonFiles = dir(fullfile(folderPath, '*.json'));
        
        if ~isempty(jsonFiles)
            % Read the first .json file
            jsonFilePath = fullfile(folderPath, jsonFiles(1).name);
            jsonData = jsondecode(fileread(jsonFilePath));
            
            % Extract the required fields: ProtocolName and InversionTime
            if isfield(jsonData, 'ProtocolName')
                ProtocolName = jsonData.ProtocolName;
            else
                warning('JSON file does not contain "ProtocolName" in folder: %s', folderPath);
                continue;
            end
            
            % If InversionTime is present, use it; otherwise, use "noInversion"
            if isfield(jsonData, 'InversionTime')
                InversionTime = jsonData.InversionTime;
            else
                InversionTime = 'noInversion';
            end

            % Create the new folder name based on ProtocolName and InversionTime
            newFolderName = sprintf('%s_TI%ss', ProtocolName, num2str(InversionTime));
            
            % Get the current subfolder's full path
            newFolderPath = fullfile(subfolders(i).folder, newFolderName);

            % Rename the subfolder if the new name is different
            if ~strcmp(folderPath, newFolderPath)
                movefile(folderPath, newFolderPath);
                fprintf('Renamed folder: %s -> %s\n', folderPath, newFolderPath);
            end
        else
            warning('No .json files found in folder: %s', folderPath);
        end
    end
end
