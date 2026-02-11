# Integrating D-OCT and Ultrasound Imaging to Determine Organ Viability - BioHack_2026
This repository presents a multimodal deep learning framework that integrates Dynamic Optical Coherence Tomography (D-OCT) with ultrasound imaging to assess kidney viability in near real time. We developed a custom embedding pipeline to fuse shallow tissue imaging from D-OCT with deep tissue imaging from ultrasound. A neural network trained on these learned embeddings predicts organ viability in under one second.

Our model can efficiently detect:
- Arteriolar fibrosis
- Interstitial fibrosis
- Tubular atrophy

The system delivers clinically relevant information comparable to biopsy, without the associated procedural risk or delay.

## Usage

To use the remade model, run the following code,

```bash
python gui.py
```

### Inputs
This will open up a new python window where you can drag in any picture you desire.

### Outputs
The output from gui.py will be an prediction of the image being a viable or non-viable kidney. In addition, it will output the time it took for the calculation.
- For the non linear result, the output it prints is the object type, but the actual object exists.

## Problem Description 

The challenge at hand was as follows:
Many viable donor organs are discarded or fail after transplant due to subjective viability assessment and inadequate preservation during transport. The lack of real-time, objective metrics, and physiologic monitoring leads to missed transplant opportunities and preventable graft failure despite ongoing organ shortages- how can we extend donor organ viability?

Kidney transplantation currently relies heavily on biopsies for viability assessment. While informative, biopsies are time-consuming, resource-intensive, and increase the risk of organ discard. Additionally, there is no widely adopted real-time tracking system for ex vivo organ evaluation. Dynamic Optical Coherence Tomography (D-OCT) has been commercially used for retinal imaging since 1996 and is well established in ophthalmology. However, it has not yet become a standard tool for kidney evaluation.

## Model Overview
In this project, we model hospital equipment rooms as suppliers and patients as the clients. If there are ```N_rooms``` clients and ```N_supply``` suppliers, then we will have matrices of ```N_supply x N_rooms``` to which we will have to apply constraints and an objective function.

### Parameters
* ```N_rooms (N_r)```: The number of patients
* ```N_supply (N_s)```: The number of supply closets
* ```flow (f)```: Matrix of flow from supply closets to patients
* ```room_supply_distance (rs)```: Matrix of distances from patient rooms to supply closets
* ```room_room_distance (rr)```: Matrix of distance from patient rooms to other patient rooms
* ```supply_supply_distance (ss)```: Matrix of distance from supply rooms to other supply rooms
* ```time_steps (T)```: Number of time steps the algorithms runs through
* ```penalty (p)```: Constant multiplier to the distance acting as a penalty of moving between time steps

### Variables
* ```X[t, i, r]```: Binary variable if at time t, patient room r occupies location i
* ```Y[t, j, s]```: Binary variable if at time t, supply room s occupies location j

### Objective
There are two parts of the objective, one from the actual objective and the other being the penalty for moving between rooms. The actual objective is given by,
```f[t, s, r] * rs[j, i] * X[(t, i, r)] * Y[(t, j, s)].``` \\
The penalty will be,
```p * rr[i, i_prev] * X[(t, i, r)] * X[(t-1, i_prev, r)] + p * ss[i, i_prev] * Y[(t, j, s)] * Y[(t-1, j_prev, s)]```.

### Constraints
There are two constraints. The first is that there is exactly 1 room/supply in each room. This is encoded by,
```sum(X[(t, i, r)] for r in range(N_rooms)) == 1```. \\
The second constraint is that each room is only chosen once over all the positions,
```sum(X[(t, i, r)] for i in range(N_rooms)) == 1```.

## Code Overview

* All of the source code is in the QAP file in the respository. If you want, you can take a look at them if needed.
* Each of the models is encoded in a function that can be called so long as you import the class in your code.
* matrix_gen.py contains all of the generation code for the random matrices.
* QAP_comparison.py in the QAP folder is a comparison platform to visualize the differences between the models.

## Results
Quantum Algorithms overall seemed to detect more optimal solutions but took longer to do so compared to a scipy algorithm. However, with a large enough sample size, we hope that, if we have enough qubits, the quantum algorithm will be more efficient.
![quantumvclassical](photos/quantumvclassical.png)
![effectiveness_rooms](photos/effectiveness_rooms.png)

## References

A. D-Wave, "D-Wave iQuHackathon 2025 Challenges", [D-Wave Challenge](https://github.com/iQuHACK/2025-D-Wave) \\
B. Ocean, "D-Wave Ocean Software Documentation", [Ocean Documentation][https://docs.ocean.dwavesys.com/en/stable/index.html]

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) file.


