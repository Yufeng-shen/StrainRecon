# StrainRecon
This software is an intra-granular strain tensor reconstruction toolkit for near-field high-energy X-ray diffraction microscopy ([nf-HEDM](https://www.andrew.cmu.edu/user/suter/3dxdm/3dxdm.html)). This readme file contains information about its [Usage](#usage), its [Dependencies](#dependencies) on other libraries, the [Structure](#package-structure) of the package, the [Formats](#file-formats) of its input and output files, and [Others](#others).

## Usage
### Simulation
### Reconstruction


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
- Directories
  - AuxData/ : outputs from standard nf-HEDM, the voxelized orientations on a cross section of the sample.
  - RealPeaks/ : outputs from MakePeakFile.ipynb, the experimental Bragg peak patterns.
  - SimResult/ : outputs from SimDemo.py, the simulated Bragg peak patterns.
  - RecResult/ : the intra-granular reconstruction results.
  - micFile/ : microstructure files from FFT simulation or regenerated from the files in AuxData/.
  - ConfigFiles/ : configure files for reconstruction or simulation.
  - util/ : some basic functions related to nf-HEDM.
  
  
## File formats

## Others
