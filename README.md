# dhSegment text

This a fork of the original [dhSegment repository](https://github.com/dhlab-epfl/dhSegment), developed to carry out experiments on combining visual and textual features published in the paper **Combining Visual and Textual Features for Semantic Segmentation of Historical Newspapers** (see reference below).


## 1. Modifications

Compared to the original dhSegment repository, the following modifications were made:

- Change of the input pipeline to read embeddings;
- Creation of embeddings maps with several dimensionality reduction algorithms;
- Concatenation of the embeddings map inside the encoder or decoder.

## 2. Usage
For general usage of dhSegment, see the [original documentation](https://dhsegment.readthedocs.io/).

- The csv file now needs four columns: image, label, embeddings, embeddings_map.
- Different configuration options were added for choosing the different hyperparameters and can be found in `dh_segment_text/utils/params_config.py` and in the encoder and decoder.
- An example config can be found under `embeddings_config.json`.

The training can be launched using the trainer script with `python dh_segment_train.py with /path/to/config.json`.

## 3. Data & Models

**Pay attention to the terms of use of the material.**

### 3.1 Data

#### Image annotations
The folder contains image annotations, with one file per newspaper containing region annotations (label and coordinates) in [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/) format (v2.0.10). 

The following licenses apply:
- `luxwort.json`: those annotations are under a [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode) license. Please refer to the right statement specified for each image in the JSON file.

- `GDL.json`, `IMP.json` and `JDG.json`: those annotations are under a [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode) license.

#### Image files
*(these files are available on Zenodo, see badge below)*
- Images of Swiss titles  (GDL, IMP, JDG) are released as an asset of the current Github [release](https://github.com/dhlab-epfl/dhSegment-text/releases/tag/0.1), in the `images.zip` archive. 
  **Terms of use**: Those images are under copyright (property of the journal *Le Temps* and of *ArcInfo*) and can be used for academic research or educational purposes only. Redistribution, publication or commercial use are not permitted. These terms of use are similar to the following right statement: http://rightsstatements.org/vocab/InC-EDU/1.0/

- Images of the Luxembourgish title are available through the IIIF endpoint of the National Library of Luxembourg (see URL in the annnotation file `luxwort.json`).  

### 3.2 Trained models
*(these files are available on Zenodo, see badge below)*

Some of the best models are released as assets of the current Github [release](https://github.com/dhlab-epfl/dhSegment-text/releases/tag/0.1) in zip files.

- **JDG_flair-FT**: this model was trained on JDG using french Flair and FastText embeddings. It is able to predict the four classes presented in the paper (`Serial`, `Weather`, `Death notice` and `Stocks`).
- **Luxwort_obituary_flair-bpemb**: this model was trained on Luxwort using multilingual Flair and Byte-pair embeddings. It is able to predict the `Death notice` class.
- **Luxwort_obituary_flair-FT_indomain**: this model was trained on Luxwort using in-domain Flair and FastText embeddings (trained on Luxwort data). It is also able to predict the `Death notice` class.

Those models can be used to predict probabilities on new images using the same code as in the original dhSegment repository.
One needs to adjust three parameters to the `predict` function: 1)  `embeddings_path` (the path to the embeddings list), 2) `embeddings_map_path`(the path to the compressed embedding map), and 3) `embeddings_dim` (the size of the embeddings).

Models are available under a [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.  
Please refer to the paper (see below) for further information or contact us.

**DOI data and models: **
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3706863.svg)](https://doi.org/10.5281/zenodo.3706863)


## 4. Paper

Please cite this paper if you are using the tool/datasets or find it relevant to your research:  

[*Combining Visual and Textual Features for Semantic Segmentation of Historical Newspapers*](https://infoscience.epfl.ch/record/282863?&ln=en). Barman Raphaël, Ehrmann Maud, Clematide Simon, Ares Oliveira Sofia, Kaplan Frédéric. 


```
@article{barman_combining_2020,
    title = {{Combining Visual and Textual Features for Semantic Segmentation of Historical Newspapers}},
    author = {Raphaël Barman and Maud Ehrmann and Simon Clematide and Sofia Ares Oliveira and Frédéric Kaplan},
    journal= {Journal of Data Mining \& Digital Humanities},
    volume= {HistoInformatics}
    DOI = {10.5281/zenodo.4065271},
    year = {2021},
    url = {https://jdmdh.episciences.org/7097},
}
```
**DOI paper:** 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4065271.svg)](https://doi.org/10.5281/zenodo.4065271)


## 5. Background & Acknowledgements

This work was carried out in the frame of the master thesis of Raphaël Barman.

We warmly thank the journal [Le Temps](https://letemps.ch) (owner of *La Gazette de Lausanne* and the *Journal de Genève*) and the group [ArcInfo](https://www.arcinfo.ch/) (owner of *L'Impartial*) for accepting to share the related datasets for academic purposes. We also thank the [National Library of Luxembourg](https://bnl.public.lu/fr.html) for its support with all steps related to the *Luxemburger Wort* annotation release.

This work was realized in the context of the ['*impresso* - Media Monitoring of the Past'](https://impresso-project.ch) project supported by the Swiss National Science Foundation under grant CR-SII5_173719.


