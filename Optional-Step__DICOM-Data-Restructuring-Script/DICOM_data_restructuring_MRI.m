
% This script calls 4 consequent functions that perform:
% 1. conversion of data to nifti
% 2. renaming of data according to sequence parameters
% 3. moving raw DICOMS o separate subfolder
% 4. creating a new filesystem by renaming each folder based on the sequence

%% Note 1: Only applies to raw DICOM data derived from Bruker Spectrometer 
% Filesystem specific
% Modifications might be required for running in other filesystems.

% Note 2: NOT NEEDED FOR THE REVIEW. THis refers to an optional part of the 
% potocol.
main_dir = 'REPLACE HERE WITH YOUR DIRECTORY'
run_dcm2niix_on_subfolders(main_dir)
rename_nifti_files_in_subfolders(main_dir)
move_dcm_files_to_subfolder(main_dir)
rename_subfolders_based_on_json(main_dir)


