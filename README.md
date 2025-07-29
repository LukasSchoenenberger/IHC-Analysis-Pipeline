# IHC-Analysis-Pipeline and IHC-MRI-Registration

Important note for reviewers

As it is not possible to grant access to the OMERO and QuPath server for external parties, we provide raw data of one brain tissue block for testing. Data can be accessed in the following link: following link:
https://filesender.switch.ch/filesender2/?s=download&token=9016f0a1-3d49-4a0d-8ca3-5931fb107d00  
Password: test_data_MS_2025!

Specifically, in the test_data.zip we provide:
    1. In sub-folder ‘../Tiles-Medium’, downloaded histology image tiles. The software will connect automatically re-attach them during the processing.
    2. In sub-folder ‘../MRI_Sequences’, acquired 3D multi-Gradient-Recalled -Echo MRI of the corresponding brain block to the histology slide in Tiles-Medium.  The folder includes the files: 
        ◦ ‘3DMGE_Magnitude_70x70x130um_TE_21p5ms_noIR.nii’ 
        ◦ ‘3DMGE_Magnitude_130x130x400um_TE_20p6ms_noIR.nii’ 
        ◦ Corresponding .json files with header information.
    3. In sub-folder ‘../Annotation_masks, .tif images, extracted from QuPath with segmentation of areas of interest.
    4. The file ‘Original_Metadata.txt’.
    5. The file ‘reference_stain_vectors.txt’.
    6. In sub-folder ‘../raw_DICOMs, raw data from two of the acquired MRI sequences for testing the optional step of restructuring the MRI acquisitions directory and converting the DICOM files to NIfTI format.



Specific additional instructions for reviewers

For testing step: Derivation of semi-quantitative maps from histological stains: Preprocessing
    • Run Sub-step 1 of this section as described in the protocol. A filesystem will be created to the selected directory.
    • Download and unzip the test data from the following link:
https://filesender.switch.ch/filesender2/?s=download&token=9016f0a1-3d49-4a0d-8ca3-5931fb107d00  
Password: test_data_MS_2025!
    • Copy and paste ‘Original_Metadata.txt’ file (../test_data/Original_Metadata.txt) and the ‘reference_stain_vectors.txt’ file (../test_data/reference_stain_vectors.txt)  into the ‘Parameters’ folder in the working directory.
    • Copy and paste the “Tiles-Medium” folder (../test_data/Tiles-Medium) into the ‘Data’ folder in the working directory.
    • Skip Sub-step 2.a.i and Sub-step 2.a.ii.
    • Run Sub-step 2.a.iii of this section as described in the protocol.
    • Skip Sub-step 2.b and Sub-step 3 of this section. These substeps would allow for downloading the histology image data and create metadata manually.
    • Continue from Sub-step 4 and follow the rest of the pipeline as described in the protocol until Sub-step 8.
    • Skip Sub-step 8.a (reference_stain_vectors file is already provided and copied in ‘Parameters’ folder).
    • Run Sub-step 8.b and follow the rest of the pipeline as described in the protocol.




For testing step: Registration of histology-derived semi-quantitative maps, histological images and histology-derived segmentations with MRI contrasts
    • Run Sub-step 1 of this section as described in the protocol. A filesystem will be created to the selected directory.
    • Download and unzip the test data from the following link:
https://filesender.switch.ch/filesender2/?s=download&token=9016f0a1-3d49-4a0d-8ca3-5931fb107d00  
Password: test_data_MS_2025!
    • Copy and paste the folder ‘MRI_Sequences’ (../test_data/MRI_Sequences) and the folder ‘Annotation_masks’ (../test_data/Annotation_masks) in your working directory.
    • Run Sub-step 2 as described in the protocol.
    • Skip Sub-step 3. This substep would allow for downloading annotation masks from QuPath.
    • Continue from Sub-step 4 and follow the rest of the pipeline as described in the protocol.

Note: the MRI data are provided in 2 resolutions (70x70x130 μm and 130x130x400μm) allowing for testing both the histology to MRI registration and the low to high resolution MRI registration. We propose using the higher resolution MRI image for the histology to MRI registration step.



For testing the OPTIONAL step: Restructuring of the MRI DICOMs in step Collecting multicontrast 9.4T MRI data
    • Download and unzip the test data from the following link:
https://filesender.switch.ch/filesender2/?s=download&token=9016f0a1-3d49-4a0d-8ca3-5931fb107d00   
Password: test_data_MS_2025!
    • Download the MATLAB scripts from the link mentioned in the Optional step: Restructuring of the MRI DICOMs. 
    • Open MATLAB.
    • Make sure that the scripts directory is added in MATLAB path, or run the following In MATLAB console, after replacing the ‘…’ with your selected path:
>> addpath('…/Optional-Step__DICOM-Data-Restructuring-Script')
    • Within the script DICOM_data_restructuring_MRI.m, replace the directory with .../test_data/raw_DICOMs (‘…’ should be replaced with your path).
    • Run the script and inspect the filesystem changes in .../test_data/raw_DICOMs/, according to the description in the protocol.


