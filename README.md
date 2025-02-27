# dune-ddm - Domain decomposition methods in Dune

### Build
This repository includes the required Dune modules as submodules.
To clone them together with this repository simly run
```bash
git clone --recursive ssh://git@parcomp-git.iwr.uni-heidelberg.de:20022/nfriess/dune-ddm.git
```
Then `cd` into the directory, create a build directory and run `cmake`:
```bash
cd dune-ddm
mkdir build
cd build
cmake ..
```
This builds all the Dune modules and the examples for `dune-ddm`.