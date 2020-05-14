# StrainRecon
This software is an intra-granular strain tensor reconstruction toolkit for near-field high-energy X-ray diffraction microscopy ([nf-HEDM](https://www.andrew.cmu.edu/user/suter/3dxdm/3dxdm.html)). This readme file contains information about its [Usage](#usage), its [Dependencies](#dependencies) on other libraries, the [Structure](#package-structure) of the package, the [Formats](#file-formats) of its input and output files.

## Dependencies
|            | version |
| ------------- | ------------- |
| CUDA       | 9.1     |
| python     | 3.6.6   |
| pycuda     | 2018.1.1|
| numpy      | 1.15.3  |
| scipy      | 1.1.0   |
| matplotlib | 3.0.0   |
| PyYAML     | 3.13    |
| h5py       |         |
| jupyter    | 1.0.0   |


## Usage

### Step 1: Peak File Generation

- Case 1: Synthetic Bragg peak patterns
    1. Generate the synthetic sample, e.g. __micFile/Ti7FFT.hdf5__
    
    2. Write the configure file, e.g. __ConfigFiles/SimG40.yml__
    
    3. Run the script for simulation: __SimDemo.py__.
       It will generate the simulated Bragg peak patterns, e.g. __SimResult/grain_40_sim.hdf5__.
       
 - Case 2: Real Bragg peak patterns
    1. Calibrate the experimental geometry parameters, e.g. __Calibration.ipynb__.
    
    2. Write the configure file, e.g. __ConfigFiles/RealG15.yml__ or __ConfigFiles/RecG40.yml__.
    
    3. Run the notebook: __MakePeakFile.ipynb__.
    It will generate an orientation and grain ID map, e.g. __micFile/Ti7HRM2nd.hdf5__, as well as a "peak file" which contains the real Bragg peak patterns of one grain, e.g. __RealPeaks/RealSample_g15.hdf5__.


### Step 2: Reconstruction

1. Reconstruct with the script __RecDemo.py__ or the notebook __Reconstruction.ipynb__.

  
## File formats
In the whole reconstruction procedure, there are four kinds of files:

- Configure file: Files in the folder ConfigFiles/. They contain the information for simulation or reconstruction, e.g. geometry parameters, sample information, file paths, etc. Its format is described in the templates.

- [Peak file](#peakfile-format): Files in the folders RealPeaks/ and SimResult/. They store the Bragg peak patterns from a single grain. They can be the output of simulation or extract from experimental images.

- [Microstructure file](#micfile-format): Files in the folder micFile/. They are the input for both reconstruction and simulation. They store the grain ID map, orientation map, and strain map (only for simulation).

- [Reconstruction file](#recfile-format): Files in the folder recFile/. They contain the reconstructed strain values.

### peakFile format
It is a hdf5 file, which stores the Bragg peak patterns in windows along with other information about the experiment. As of now, the window size is fixed as (&Delta;J=300, &Delta;K=160, &Delta;&Omega;=45), the units are number of pixels, number of pixels, and number of frames. The datasets are: (assuming there are N+1 peaks recorded)

- "/Gs": shape of (N+1,3). The corresponding reciprocal vectors before distortion. 

- "/MaxInt": shape of (N+1). The maximum intensities of peaks.

- "/OrienM": shape of (3,3). The average orientation of the grain.

- "/Pos": shape of (3). The center of mass of the grain.

- "/avg_distortion": shape of (3,3). The strain already considered in "/Gs".

- "/limits": shape of (N+1,5). The pixel coordinates of the window and the &Omega; indices of the first frame.

- "/whichOmega": shape of (N+1). Indicate is the first or second Bragg peak of that reciprocal vectors.

- "/Imgs/Im0": shape of (160,300,45). The diffraction pattern of the first Bragg peak.
...
- "/Imgs/ImN": shape of (160,300,45). The diffraction pattern of the last Bragg peak.


### micFile format
It is a hdf5 file, which contains following datasets: (assuming the mesh on sample cross section  has N<sub>x</sub> columns and N<sub>y</sub> rows)

- "/origin": shape of (2). The position (x,y) of the bottom left corner of the mesh, in the unit of millimeter.

- "/stepSize": shape of (2). The step size (&delta;x,&delta;y) of the mesh.

- "/Confidence": shape of (N<sub>y</sub>,N<sub>x</sub>). The confidence from I9 reconstruction.

- "/Ph1": shape of (N<sub>y</sub>,N<sub>x</sub>). The first Euler angles on the voxels.

- "/Psi": shape of (N<sub>y</sub>,N<sub>x</sub>). The second Euler angles on the voxels.

- "/Ph2": shape of (N<sub>y</sub>,N<sub>x</sub>). The third Euler angles on the voxels.

- "/GrainID": shape of (N<sub>y</sub>,N<sub>x</sub>). The grain IDs on the voxels.

We also recommend store the following two datasets:

- "/Xcoordinate": shape of (N<sub>y</sub>,N<sub>x</sub>). The x coordinates of the voxels, in the unit of millimeter.

- "/Ycoordinate": shape of (N<sub>y</sub>,N<sub>x</sub>). The y coordinates of the voxels, in the unit of millimeter.

To be used for simulation, following datasets for the elastic strain components are also needed:

- "/E11": shape of (N<sub>y</sub>,N<sub>x</sub>).

- "/E12": shape of (N<sub>y</sub>,N<sub>x</sub>).

- "/E13": shape of (N<sub>y</sub>,N<sub>x</sub>).

- "/E22": shape of (N<sub>y</sub>,N<sub>x</sub>).

- "/E23": shape of (N<sub>y</sub>,N<sub>x</sub>).

- "/E33": shape of (N<sub>y</sub>,N<sub>x</sub>).

### recFile format
It is a hdf5 file, which contains five important datasets: (assuming there are N voxels in the grain)

- "/Phase2_S": shape of (N,3,3). The reconstructed distortion matrix in reciprocal space on voxels. (matrix __D__ in my thesis, p.g. 56)

- "/realS": shape of (N,3,3). The reconstructed strain tensor on voxels. (matrix __V__ in my thesis, p.g. 55)

- "/realO": shape of (N,3,3). The reconstructed orientation matrices on voxels. (matrix __R__ in my thesis, p.g. 55)

- "/x": shape of (N). The x coordinates of voxels, in the unit of millimeter.

- "/y": shape of (N). The y coordinates of voxels, in the unit of millimeter.

Other information about the reconstruction may also installed.

## Package Structure
- Basics
    - strain_device.cu : CUDA kernel functions.
    - config.py : wrapper for configuration files.
    - initializer.py : constructs GPU related functions and objects, along with the Detector and Material objects.
    - simulator.py : simulates the Bragg peak patterns from a synthetic data, it can be used for validating the reconstruction algorithm.
    - reconstructor.py : reconstructs the intra-granular strain tensor from Bragg peak patterns.
    - optimizers.py : the minimization algorithms used in the reconstruction.
    
- Scripts
    - Calibration.ipynb : calibrates the geometry parameters in the experimental setup.
    - MakePeakFile.ipynb : extracts the windows around the Bragg peaks from the grain we want to reconstruct. 
    - SimDemo.py : the script to simulate the Bragg peak patterns from a synthetic data, it can be used for validating the reconstruction algorithm.
    - RecDemo.py : the script to reconstruct the intra-granular strain tensor from Bragg peak patterns.
    - Reconstruction.ipynb : an alternative and more flexible way to do reconstruction instead of RecDemo.py, basically contains the procedures in reconstructor.py.
    
- Folders
    - AuxData/ : outputs from standard nf-HEDM, the voxelized orientations on a cross section of the sample.
    - RealPeaks/ : outputs from MakePeakFile.ipynb, the experimental Bragg peak patterns.
    - SimResult/ : outputs from SimDemo.py, the simulated Bragg peak patterns.
    - RecResult/ : the intra-granular reconstruction results.
    - micFile/ : microstructure files from FFT simulation or regenerated from the files in AuxData/.
    - ConfigFiles/ : configure files for reconstruction or simulation.
    - util/ : some basic functions related to nf-HEDM.
  



