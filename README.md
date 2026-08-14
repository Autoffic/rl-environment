# traffic-intersection-rl-environment-sumo

## Meeting Requirements
`pip install -r requirements.txt`

## installing environment
- `cd custom_gym`

- `pip install -e .`

- `cd ..`

## Setup Environment
`source ./setup.sh`

## Start training
`python ./scripts/train.py`

## See the results
`python ./scripts/traffic.py`

## For gpu support
You need to uninstall pytorch cpu version if installed already and install the gpu version.
https://pytorch.org/get-started/locally/

### Problem with requirements.txt ?
You may change the version or the name of the dependency to match your platform.