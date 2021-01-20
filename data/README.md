## Combining textual and visual features for newspaper article segmentation: Datasets & Models



### Image annotations
The folder contains image annotations, with one file per newspaper containing region annotations (label and coordinates) in [VIA](http://www.robots.ox.ac.uk/~vgg/software/via/) format (v2.0.10). 

The following licenses apply:
- `luxwort.json`: those annotations are under a [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode) license. Please refer to the right statement specified for each image in the JSON file.

- `GDL.json`, `IMP.json` and `JDG.json`: those annotations are under a [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode) license.

  

### Image files
- Images of Swiss titles  (GDL, IMP, JDG) are released as an asset of the current Github [release](https://github.com/dhlab-epfl/dhSegment-text/releases/tag/0.1), in the `images.zip` archive. 
  **Terms of use**: those images are under copyright but can be used for research purposes only. Redistribution, publication or commercial use are ***not*** permitted. 

- Images of the Luxembourgish title are available through the IIIF endpoint of the National Library of Luxembourg (see URL in the annnotation file `luxwort.json`).  

  

### Trained models

Some of the best models are released as assets of the current Github [release](https://github.com/dhlab-epfl/dhSegment-text/releases/tag/0.1) in zip files.

- **JDG_flair-FT**: this model was trained on JDG using french Flair and FastText embeddings. It is able to predict the four classes presented in the paper (`Serial`, `Weather`, `Death notice` and `Stocks`).
- **Luxwort_obituary_flair-bpemb**: this model was trained on Luxwort using multilingual Flair and Byte-pair embeddings. It is able to predict the `Death notice` class.
- **Luxwort_obituary_flair-FT_indomain**: this model was trained on Luxwort using in-domain Flair and FastText embeddings (trained on Luxwort data). It is also able to predict the `Death notice` class.

Those models can be used to predict probabilities on new images using the same code as in the original repository.
One needs to adjust three parameters to the `predict` function: 1)  `embeddings_path` (the path to the embeddings list), 2) `embeddings_map_path`(the path to the compressed embedding map), and 3) `embeddings_dim` (the size of the embeddings).

Models are available under a [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.  Please refer to the [paper](https://github.com/dhlab-epfl/dhSegment-text#paper) for further information or contact us.



### DOI

[https://doi.org/10.5281/zenodo.3706863](https://doi.org/10.5281/zenodo.3706863)




## Acknowledgements

We warmly thank the journal [Le Temps](https://letemps.ch) (owner of *La Gazette de Lausanne* and the *Journal de Genève*) and the group [ArcInfo](https://www.arcinfo.ch/) (owner of *L'Impartial*) for accepting to share the related datasets for academic purposes. We also thank the [National Library of Luxembourg](https://bnl.public.lu/fr.html) for its support with all steps related to the *Luxemburger Wort* annotation release.



  



