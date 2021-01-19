# dhSegment text

This a fork of the original [dhSegment repository](https://github.com/dhlab-epfl/dhSegment), developed to carry out experiments on combining visual and textual features (see paper reference below).


## Modifications

The following modifications were made:

- Changing the input pipeline to read embeddings;
- Creation of embeddings maps with several dimensionality reduction algorithms;
- Concatenation of the embeddings map inside the encoder or decoder.

## Usage
For general usage of dhSegment, see the [original documentation](https://dhsegment.readthedocs.io/).

- The csv file now needs four columns: image, label, embeddings, embeddings_map.
- Different configuration options were added for choosing the different hyperparameters and can be found in `dh_segment_text/utils/params_config.py` and in the encoder and decoder.
- An example config can be found under `embeddings_config.json`.

The training can be launched using the trainer script with `python dh_segment_train.py with /path/to/config.json`.

## Data & Models

Datasets and best models of experiments with newspaper article segmentation are available in the `data` folder. 
See this [README](https://github.com/dhlab-epfl/dhSegment-text/tree/master/data).

**Pay attention to the terms of use of the material.**

## Paper

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

## Background

This work was carried out in the frame of the master thesis of Raphaël Barman, within the context of the ['*impresso* - Media Monitoring of the Past'](https://impresso-project.ch) project (SNSF Sinergia grant [CRSII5_173719](http://p3.snf.ch/project-173719)).


