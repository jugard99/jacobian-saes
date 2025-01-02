# Jacobian SAEs
The goal of this paper is to create something similar to SAEs but which optimizes for _sparsity of computation_ rather than merely sparsity of internal representations.

This is an ongoing project and is still under development; I'll write a more comprehensive readme once the paper is ready for publication (likely around the ICML deadline).

## Credit
The code base started off as a fork of the excellent [**SAELens**](https://github.com/jbloomAus/SAELens/tree/main), and a lot of credit goes to them for giving me such a great starting point.

## How to run it
The top-level runner is `runners/train_jacobian_sae.py`.

### If you're an internal collaborator at the University of Bristol
There are some bash scripts in `hpc_utils` that you might find useful
- You can use the `run` bash script to run the code on the BluePebble cluster (note that you'll need to run this one on the remote rather than locally)
- You can use `copy` to copy your local files onto BluePebble
- You can use `bp_run` to copy the files and then run them

The last two assume you have a remote called `bp` in your `.ssh/config`, and all of them assume you've cloned [Laurence's infrastructure repo](https://github.com/LaurenceA/infrastructure) onto BluePebble and that it's on your PATH.
