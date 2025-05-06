# Seg_3D_by_PC2D: Multi-View Projection for Domain Generalization and Adaptation in 3D Semantic Segmentation

Official PyTorch implementation of the method **Seg_3D_by_PC2D**. More details can be found in the paper:

**Seg_3D_by_PC2D: Multi-View Projection for Domain Generalization and Adaptation in 3D Semantic Segmentation**, (preprint) [[arXiv to come](https://github.com/andrewcaunes/ia4markings)]
by *Andrew Caunes, Thierry Chateau, Vincent Frémont*

![Overview of the method](./imgs/overview.pdf)

## Code Availability

The full codebase will be made publicly available upon the publication of the paper. Until then, this repository serves as a preliminary source to detail the setup and parameters used in our experiments.

## Implementation details

[classes_dicts.py](./classes_dicts.py) provides utilities and dictionaries to map the classes of nuScenes and SemanticKITTI.
Example of usage:
This will print infos on the mapping between the DGLSS classes system
used for Domain Generalization experiments and the classes of the SemanticKITTI dataset.
```bash
python3 classes_dicts --cs1 semantickitti_dglss --cs2 dglss
```
other classes systems include:
- `uda`: the classes system used for the Unsupervised Domain Adaptation experiments (includind `manmade')
- `semantickitti`: the original classes of the SemanticKITTI dataset, ready to be mapped to the UDA classes system
- `nuscenes`: the classes of the nuScenes dataset
- `dglss`: the DGLSS classes system
- `semantickitti_dglss`: the classes of the SemanticKITTI dataset 
ready to be mapped to the DGLSS classes system
- `nuscenes_dglss`: the classes of the nuScenes dataset ready to be mapped to the DGLSS classes system