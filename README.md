[![Article DOI:10.1109/JSTARS.2024.3480520](https://img.shields.io/badge/Article%20DOI-10.1109%2FJSTARS.2024.3480520-blue)](https://doi.org/10.1109/JSTARS.2024.3480520)

# [DTACSNet: Onboard Cloud Detection and Atmospheric Correction With Efficient Deep Learning Models](https://ieeexplore.ieee.org/document/10716772)

*Cesar Aybar*<sup>§</sup>, *Gonzalo Mateo-García*<sup>§</sup>, *Giacomo Acciarini*<sup>§</sup>, *Vit Ruzicka*, *Gabriele Meoni*, *Nicolas Longepe*, *Luis Gómez-Chova*

<sub><sup>*§ development contribution*</sup></sub>

This repo contains an open implementation to run inference with DTACSNet models for atmospheric correction. The trained models
provided here are customized to the band configuration that will be available in Phi-Sat-II. **This repo and trained models are released under a** [Creative Commons non-commercial licence](https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt) 
<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc.png" alt="licence" width="60"/>

Install ⚙️:
```bash
pip install dtacs
```

<img src="assets/example_ac.png" alt="awesome atmospheric correction" width="100%">
The figure above shows a sample of Sentinel-2 level 1C, DTACSNet model output and Sentinel-2 level 2A in the RGB (first row) and in the SWIR, NIR, Red (last row) composites.

## Acknowledgments

DTACSNet has been developed by Trillium Technologies. It has been funded by ESA Cognitive Cloud Computing in Space initiative project number D-TACS I-2022-00380.

## Citation

If you find this work useful for your research, please consider citing [our work](https://ieeexplore.ieee.org/document/10716772):

```bibtex
@article{aybar_onboard_2024,
	title = {Onboard {Cloud} {Detection} and {Atmospheric} {Correction} {With} {Efficient} {Deep} {Learning} {Models}},
	volume = {17},
	issn = {2151-1535},
	url = {https://ieeexplore.ieee.org/abstract/document/10716772},
	doi = {10.1109/JSTARS.2024.3480520},
	urldate = {2024-11-12},
	journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
	author = {Aybar, Cesar and Mateo-García, Gonzalo and Acciarini, Giacomo and Růžička, Vít and Meoni, Gabriele and Longépé, Nicolas and Gómez-Chova, Luis},
	year = {2024},
	note = {Conference Name: IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
	pages = {19518--19529}
}
```

## More Cloud Detection Viz

![#8db5f0](https://placehold.co/15x15/8db5f0/8db5f0.png) Thick cloud
![#8df094](https://placehold.co/15x15/8df094/8df094.png) Thin cloud 
![#fff982](https://placehold.co/15x15/fff982/fff982.png) Cloud shadow

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/OfuPpQGUqOFyqB26YrlDF.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/5d0nbfVGvUQHuzNUKVqq5.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/b0I0ZAgFwsbUoJOLw5z6E.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/dZlcpUVXi6XJ7Xb0Ig9fm.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/phuSxN81fwl9oP-nck4av.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/ESof9Ota75fTgsT0sYNnl.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/TmKdQ6zwtZnD2xFBi-Jvf.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/-IE4TW5cjrKCOmbI0nq9w.png)

## More Atmospheric Correction Viz

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/t_8CiBDUqBdafIV9w4ylK.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/JMyEEYn3aMJZrz3BDfvXs.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/-8oo6ke6GgRvaadsVrviq.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/fSue-_WxTla3IRH5VplGJ.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/STyfQtbNkdLEx-HBI3V-V.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/RvXFiBDUjd4wQcjz8pSEH.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/IiGIz-W8KMsxuMPeM3Ogh.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6402474cfa1acad600659e92/PNco-ihWwqSSLICgOtKsB.png)

