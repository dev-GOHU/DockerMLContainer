# OS
default: ubuntu22.04

# Installed Packages
- CUDA (Default 11.8)
- cuDNN (Default 8.6)
- tensorRT (Not Installed)
- tensorflow (Default 2.13.0)
- pytorch

##
You can change package's version by modifing Dockerfile ENV before docker building.
But if cudnn major version is upper 8, you must modify lines that install cudnn.