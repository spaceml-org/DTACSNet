# DTACSNet: onboard cloud detection and atmospheric correction with end-to-end deep learning emulators

*Giacomo Acciarini*<sup>§</sup>, *Cesar Aybar*<sup>§</sup>, *Gonzalo Mateo-García*<sup>§</sup>, *Vit Ruzicka*, *Atilim Gunes Baydim*, *Gabriele Meoni*, *Nicolas Longepe*, *Luis Gómez-Chova*

<sub><sup>*§ equal contribution*</sup></sub>

This repo contains an open implementation to run inference with DTACSNet models for atmospheric correction. The trained models
provided here are customized to the band configuration that will be available in Phi-Sat-II. **This repo and trained models are released under a** [Creative Commons non-commercial licence](https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt) 
<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc.png" alt="licence" width="60"/>

See the [inference tutorial](./tutorials/inference.ipynb) for an example of running the model.

<img src="example_ac.png" alt="awesome atmospheric correction" width="100%">
The figure above shows a sample of Sentinel-2 level 1C, atmospheric correction model output and Sentinel-2 level 2A in the RGB (first row) and in the SWIR, NIR, Red (last row) composites.

## Acknowledgments

DTACSNet has been developed by Trillium Technologies. It has been funded by ESA Cognitive Cloud Computing in Space initiative project number D-TACS I-2022-00380.
