# dhSegment text

This a fork of the original [dhSegment repository](https://github.com/dhlab-epfl/dhSegment), developed to carry out experiments on combining visual and textual features (see paper reference below).




## Modifications

The following modifications were made:

- Changing the input pipeline to read embeddings
- Creation of embeddings maps with several dimensionality reduction algorithms
- Concatenation of the embeddings map inside the encoder or decoder

## Usage
For general usage of dhSegment, see the [original documentation](https://dhsegment.readthedocs.io/).

- The csv file now needs four columns: image, label, embeddings, embeddings_map.
- Different configuration options were added for choosing the different hyperparamters and can be found in `dh_segment_text/utils/params_config.py` and in the encoder and decoder.
- An example config can be found under `embeddings_config.json`.

The training can be launched using the trainer script with `python dh_segment_train.py with /path/to/config.json`.

## Data & Models
Datasets and best models of experiments with newspaper article segmentation are available in the `data` folder. 
Pay attention to the terms of use of the material.

## Paper
Barman Raphaël, Ehrmann Maud, Clematide Simon, Ares Oliveira Sofia, Kaplan Frédéric (to appear in Autumn 2020). *Combining Visual and Textual Features for Semantic Segmentation of Historical Newspapers*, Journal of Data Mining and Digital Humanities. [https://arxiv.org/abs/2002.06144](https://arxiv.org/abs/2002.06144)

**Bibtex** (will be sooon updated)
```
@article{barman_combining_2020,
    title = {{Combining Visual and Textual Features for Semantic Segmentation of Historical Newspapers}},
    author = {Raphaël Barman and Maud Ehrmann and Simon Clematide and Sofia Ares Oliveira and Frédéric Kaplan},
    journal= {Journal of Data Mining and Digital Humanities},
    volume= {tbu}
    month = {tbu} 
    year = {2020},
    eprint = {2002.06144},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
}
```
