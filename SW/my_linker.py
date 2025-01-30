import sys
import os
import re

import lib

import torch
import numpy as np

import shutil

import networkx as nx

try:
    _, image_path, segmentation_path, output_path = sys.argv
except ValueError:
    print("Usage: python tracker.py <image_path> <segmentation_path> <output_path>")
    sys.exit(1)

path = os.path.normpath(output_path)
dataset = path.split(os.sep)[-2]  # 'DATASET'

# Load images and segmentations
images, digits = lib.load_images(image_path)
segmentations, _ = lib.load_images(segmentation_path)
print(f"Loaded {len(images)} images and {len(segmentations)} segmentations.")


# Load model
model = lib.load_model(dataset)
print("Model successfully loaded.")

# Build graph
if dataset =='BF-C2DL-HSC':
    radius = 0.02
elif dataset == 'Fluo-N2DL-HeLa':
    radius = 0.05
elif dataset == 'PhC-C2DL-PSC':
    radius = 0.04
else: radius = 0.2

graph = lib.build_graph(segmentations, radius)
graph = graph.to("cpu")
# graph = graph.to("cuda" if torch.cuda.is_available() else "cpu")

# # TODO: Only for validation purposes
# model.eval()
# probs = model(graph)
# predictions = probs.cpu().detach().numpy() > 0.5

# TODO: Only for validation purposes
model.eval()
with torch.no_grad():
    probs = model(graph)
    predictions = probs.cpu().detach().numpy() > 0.5

# graph = graph.to("cpu")
coords = np.array(graph.x)
coords[:, 0] *= images[0].shape[0]
coords[:, 1] *= images[0].shape[1]

frame = np.array(graph.frames, dtype=int)
edge_index = np.array(graph.edge_index, dtype=int)
predictions = np.array(predictions, dtype=int)
probabilities = probs.cpu().detach().numpy()


SEQID = [digit for digit in path.split(os.sep)[-1] if digit.isdigit()]  
SEQID = "".join(SEQID)
temp_path = f"./{dataset}_{SEQID}_temp"

for path in [output_path, temp_path]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


G = nx.DiGraph()
for i, pos in enumerate(coords):
    G.add_node(i, position=pos, frame=frame[i])

# Add edges with predictions
for i in range(edge_index.shape[1]):
    sender, receiver = edge_index[:, i]
    if predictions[i] == 1:  # Add edge only if prediction indicates a connection
        G.add_edge(sender, receiver, probability=probabilities[i])

pruned_G = lib.prune_graph(G)


## remove multiple parenthood
G_processed = pruned_G.copy()
for receiver in pruned_G.nodes:
    incoming_edges = list(pruned_G.in_edges(receiver, data=True))

    if len(incoming_edges) > 1:
        # Sort the incoming edges by the 'probability' attribute in descending order
        incoming_edges_sorted = sorted(
            incoming_edges, key=lambda x: x[2]["probability"], reverse=True
        )

        # Keep the edge with the highest probability, remove the others
        for sender, receiver, data in incoming_edges_sorted[1:]:
            G_processed.remove_edge(sender, receiver)

lib.extract_trajectories(
    G_processed, segmentation_path, output_path, temp_path, digits=digits
)
# delete the temp folder
shutil.rmtree(temp_path)
