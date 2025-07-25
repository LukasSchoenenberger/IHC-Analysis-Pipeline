
function rename_nifti_files_in_subfolders(motherFolder)
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

        % Look for the .json files in the subfolder
        jsonFiles = dir(fullfile(folderPath, '*.json'));
        
        % Process each .json file found in the folder
        for j = 1:length(jsonFiles)
            jsonFilePath = fullfile(folderPath, jsonFiles(j).name);
            jsonData = jsondecode(fileread(jsonFilePath));
            
            % Check for required fields in the .json file
            if isfield(jsonData, 'ProtocolName')
                ProtocolName = jsonData.ProtocolName;
                
                % Extract InversionTime or use "noInversion" if it doesn't exist
                if isfield(jsonData, 'InversionTime')
                    InversionTime = jsonData.InversionTime;
                else
                    InversionTime = 'noInversion';
                end
                
                % Extract EchoTime or use "noEchoTime" if it doesn't exist
                if isfield(jsonData, 'EchoTime')
                    EchoTime = jsonData.EchoTime;
                else
                    EchoTime = 'noEchoTime';
                end
                
                % Find the corresponding .nii file based on the json file name
                niiFileName = strrep(jsonFiles(j).name, '.json', '.nii');
                niiFilePath = fullfile(folderPath, niiFileName);
                
                if isfile(niiFilePath)
                    % Generate new names for the .nii and .json files
                    newNiiName = sprintf('%s_TI%ss_TE%ss.nii', ProtocolName, num2str(InversionTime), num2str(EchoTime));
                    newJsonName = sprintf('%s_TI%ss_TE%ss.json', ProtocolName, num2str(InversionTime), num2str(EchoTime));
                    
                    % Check if the new file names are different from the old ones before renaming
                    newNiiFilePath = fullfile(folderPath, newNiiName);
                    if ~strcmp(niiFilePath, newNiiFilePath)
                        movefile(niiFilePath, newNiiFilePath);
                        fprintf('Renamed NIfTI file: %s -> %s\n', niiFilePath, newNiiFilePath);
                    end
                    
                    % Rename the corresponding .json file
                    newJsonFilePath = fullfile(folderPath, newJsonName);
                    if ~strcmp(jsonFilePath, newJsonFilePath)
                        movefile(jsonFilePath, newJsonFilePath);
                        fprintf('Renamed JSON file: %s -> %s\n', jsonFilePath, newJsonFilePath);
                    end
                else
                    warning('No corresponding NIfTI file found for JSON file: %s', jsonFilePath);
                end
            else
                warning('JSON file does not contain the expected field "ProtocolName" in file: %s', jsonFilePath);
            end
        end
    end
end
