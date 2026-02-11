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


## Code Overview


## Results

