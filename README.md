# DTACSNet: onboard cloud detection and atmospheric correction with end-to-end deep learning emulators

*Cesar Aybar*<sup>§</sup>, *Gonzalo Mateo-García*<sup>§</sup>, *Giacomo Acciarini*<sup>§</sup> *Vit Ruzicka*, *Gabriele Meoni*, *Nicolas Longepe*, *Luis Gómez-Chova*

<sub><sup>*§ equal contribution*</sup></sub>

This repo contains an open implementation to run inference with DTACSNet models for atmospheric correction. The trained models
provided here are customized to the band configuration that will be available in Phi-Sat-II. **This repo and trained models are released under a** [Creative Commons non-commercial licence](https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt) 
<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc.png" alt="licence" width="60"/>

See the [inference tutorial](https://github.com/spaceml-org/DTACSNet/blob/main/tutorials/inference_Sentinel-2.ipynb) for an example of running the model.

<img src="example_ac.png" alt="awesome atmospheric correction" width="100%">
The figure above shows a sample of Sentinel-2 level 1C, DTACSNet model output and Sentinel-2 level 2A in the RGB (first row) and in the SWIR, NIR, Red (last row) composites.

## Acknowledgments

DTACSNet has been developed by Trillium Technologies. It has been funded by ESA Cognitive Cloud Computing in Space initiative project number D-TACS I-2022-00380.

## Citation

If you find this work useful for your research, please consider citing:

```
@inproceedings{mateo-garcia_onboard_2023,
	title = {Onboard {Cloud} {Detection} and {Atmospheric} {Correction} with {Deep} {Learning} {Emulators}},
	url = {https://ieeexplore.ieee.org/document/10282605},
	doi = {10.1109/IGARSS52108.2023.10282605},
	booktitle = {{IGARSS} 2023 - 2023 {IEEE} {International} {Geoscience} and {Remote} {Sensing} {Symposium}},
	author = {Mateo-García, Gonzalo and Aybar, Cesar and Acciarini, Giacomo and Růžička, Vít and Meoni, Gabriele and Longépé, Nicolas and Gómez-Chova, Luis},
	month = jul,
	year = {2023},
	note = {ISSN: 2153-7003},
	pages = {1875--1878}
}
```
