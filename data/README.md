## About:

**Datasets and models** related to the experiments on combining textual and visual features for newspaper article segmentation.



**Zenodo record:** (upcoming) 10.5281/zenodo.4065271

### Annotations
The folder contains image annotations, with one file per newspaper containing region annotations (label and coordinates) in [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/) format. Depending on the newspaper, those annotations are under license [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) or [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Please refer to the rights statements in each file.
### Images
The images are released as an asset to the current Github release in a zip file. It contains the files of the Swiss titles (GDL, IMP, JDG). Those images are under copyright but can be used for research purposes (redistribution, publication or commercial use are ***not*** permitted). Images of the Luxembourgish title are available through the IIIF endpoint of the National Library of Luxembourg.  Please refer to the rights statements and information in each file.

### trained-models

The models are released as assets of the current Github release in corresponding zip files. They contains some of the best models, as described in the corresponding paper. Available under a [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

They can be used to predict probabilities on new images using the same code as in the original repository.
The only difference is the need for three additional parameters to the `predict` function, `embeddings_path` which the path to the embeddings list,
`embeddings_map_path` which is the path to the compressed embedding map and `embeddings_dim` which is the size of the embeddings.

The following models are shared:
- **JDG_flair-FT**: trained on JDG using french Flair and FastText embeddings and is able to predict
the four classes presented in the paper (`Serial`, `Weather`, `Death notice` and `Stocks`).
- **Luxwort_obituary_flair-bpemb**: trained on Luxwort using multilingual Flair and Byte-pair embeddings and is able to predict only the `Death notice` class.
- **Luxwort_obituary_flair-FT_indomain**: trained on Luxwort using in-domain Flair and FastText embeddings (trained on Luxwort data) and is also only able to
predict the `Death notice` class.


## Acknowledgements:

We warmly thank the journal [Le Temps](https://letemps.ch) (owner of *La Gazette de Lausanne* and the *Journal de Genève*) and the group [ArcInfo](https://www.arcinfo.ch/) (owner of *L'Impartial*) for accepting to share the related datasets for academic purposes. We also thank the [National Library of Luxembourg](https://bnl.public.lu/fr.html) for its work and support with all steps related to the *Luxemburger Wort* annotation release.

  



