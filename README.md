# SoftAgent
This repository is built based on [SoftGym](https://github.com/Xingyu-Lin/softgym), with modified action space and added new testing environments and action space reduction method. 
The benchmarked tasks include
* cem\_rope\_straightening [[source](./cem_rope_straightening)]  
* cem\_rope\_folding [[source](./cem_rope_folding)]
* cem\_cloth\_diagonal\_folding [[source](./cem_diagonal_folding)]
* cem\_cloth\_side\_folding [[source](./cem_cloth_side_folding)]
* cem\_cloth\_reflective\_folding [[source](./cem_cloth_reflective_folding)]
* cem\_cloth\_underneath\_folding [[source](./cem_cloth_underneath_folding)]
## Installation 

1. Install SoftGym by following the instructions in [SoftGym](https://github.com/Xingyu-Lin/softgym) repository. Then, copy the softgym code to the SoftAgent root directory so we have the following file structure:
    ```
    softagent
    ├── cem
    ├── ...
    ├── softgym
    ```
2. Update conda env with additional packages required by SoftAgent: `conda env update  --file environment.yml  --prune`
3. Activate the conda environment by running `. ./prepare_1.0.sh`.

## Running benchmarked experiments 

1. Running rope straightening experiments: `cd cem_rope_straightening; python run_cem_rope_straightening.py`. Refer to `run_cem_rope_straightening.py` for different arguments.
2. Running rope straightening experiments: `cd cem_rope_folding; python run_cem_rope_folding.py`. Refer to `run_cem_rope_folding.py` for different arguments.
3. Running rope straightening experiments: `cd cem_cloth_diagonal_folding; python run_cem_cloth_diagonal_folding.py`. Refer to `run_cem_cloth_diagonal_folding.py` for different arguments.
4. Running rope straightening experiments: `cd cem_cloth_side_folding; python run_cem_cloth_side_folding.py`. Refer to `run_cem_cloth_side_folding.py` for different arguments.
5. Running rope straightening experiments: `cd cem_cloth_reflective_folding; python run_cem_cloth_reflective_folding.py`. Refer to `run_cem_cloth_reflective_folding.py` for different arguments.
6. Running rope straightening experiments: `cd cem_cloth_underneath_folding; python run_cem_cloth_underneath_folding.py`. Refer to `run_cem_cloth_underneath_folding.py` for different arguments.

<!-- ### PyFleX APIs
Please see the example test scripts and the bottom of `bindings/pyflex.cpp` for available APIs. -->

## Cite
If you find this codebase useful in your research, please consider citing:
```
@inproceedings{icra2023arsoftagent,
 title={Goal-Conditioned Action Space Reduction for Deformable Object Manipulation},
 author={Wang, Shengyin and Papallas, Rafael and Leonetti, Matteo and Dogar, Mehmet},
 booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
 year={2023}
}
```

## References
- softagent repository: https://github.com/Xingyu-Lin/softagent
- Softgym repository: https://github.com/Xingyu-Lin/softgym

