# MAGIK - A Geometric Deep Learning Framework

## Introduction
MAGIK is a geometric deep-learning framework for analyzing biological system dynamics from time-lapse microscopy. It models object movement and interactions through a directed graph, where nodes represent object detections at specific times, and edges connect nodes that are spatiotemporally close. The framework aims to prune redundant edges while retaining true connections using a Message Passing Neural Network.

## Features
- Graph-based representation of object trajectories
- Flexible feature encoding for nodes and edges
- Edge classification for trajectory reconstruction
- Postprocessing algorithm to refine connections

## Execution Details
A Python Notebook is provided for training and applying MAGIK on 2D datasets from the Cell Tracking Challenge. The notebook is divided into five main sections:
1. **Reading and Viewing the Data** - Download and visualize datasets.
2. **Graph Construction** - Build a directed spatiotemporal graph from segmentation maps.
3. **Dataset Construction** - Generate training data using stochastic sampling.
4. **MAGIK Definition and Training** - Define and train MAGIK using the deeplay deep learning package.
5. **Model Evaluation** - Assess prediction quality and visualize trajectories.

## References
Pineda, J., Midtvedt, B., Bachimanchi, H., Noé, S., Midtvedt, D., Volpe, G., & Manzo, C. (2023). Geometric deep learning reveals the spatiotemporal features of microscopic motion. *Nature Machine Intelligence, 5*, 71-82.

## BibTeX
```bibtex
@article{pineda2023geometric,
  author = {Pineda, J. and Midtvedt, B. and Bachimanchi, H. and Noé, S. and Midtvedt, D. and Volpe, G. and Manzo, C.},
  title = {Geometric deep learning reveals the spatiotemporal features of microscopic motion},
  journal = {Nature Machine Intelligence},
  volume = {5},
  pages = {71-82},
  year = {2023}
}
```
