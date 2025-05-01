## Installation
Clone the GitHub repository with
```
git clone https://github.com/UMThorntonGroup/Al3Ti_ECP.git
```
Use whatever python environment manager you would like (venv, conda, etc...). I prefer using Python virtual environments, which can be created with
```
python3 -m venv al3ti_ecp
```
and activated with
```
source al3ti_ecp/bin/activate
```
This script requires a few python packages that can installed with the following commands.
```
python3 -m pip install --upgrade pip
pip install pycalphad matplotlib numpy
```
The code can then be run with
```
python3 nucleation_rate.py
```