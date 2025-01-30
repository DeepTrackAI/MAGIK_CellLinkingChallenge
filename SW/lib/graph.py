from scipy.spatial import distance

import numpy as np
from skimage import measure

import torch
from torch_geometric.data import Data


class GraphFromSegmentations:
    """
    Convert a list of segmentations into a graph dataset.

    Parameters
    ----------
    connectivity_radius : float
        Maximum distance between nodes to create an edge.
    max_frame_distance : int
        Maximum number of frames between nodes to create an edge.

    Returns
    -------
    torch_geometric.data.Data
        Graph dataset.
    """

    def __init__(self, connectivity_radius, max_frame_distance):
        self.connectivity_radius = connectivity_radius
        self.max_frame_distance = max_frame_distance

    def __call__(self, segmentations):
        x, node_index_labels, frames = [], [], []
        for frame, segmentation in enumerate(segmentations):
            features, index_labels = self.compute_node_features(segmentation)
            x.append(features)
            node_index_labels.append(index_labels)
            frames.append([frame] * len(features))

        x = np.concatenate(x)
        node_index_labels = np.concatenate(node_index_labels)
        frames = np.concatenate(frames)

        edge_index, edge_attr = self.compute_connectivity(x, frames)
        # edge_ground_truth = self.compute_ground_truth(
        #     node_index_labels, edge_index, relation
        # )

        return Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index.T, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr[:, None], dtype=torch.float),
            distance=torch.tensor(edge_attr[:, None], dtype=torch.float),
            frames=torch.tensor(frames, dtype=torch.float),
            # y=torch.tensor(edge_ground_truth[:, None], dtype=torch.float),
        )

    def compute_node_features(self, segmentation):
        labels = np.unique(segmentation)[1:]
        return zip(
            *[
                (
                    measure.regionprops(
                        (segmentation == label).astype(int),
                    )[0].centroid
                    / np.array(segmentation.shape),
                    label,
                )
                for label in labels
            ]
        )

    def compute_connectivity(self, x, frames):
        positions = x[:, :2]
        distances = distance.squareform(distance.pdist(positions, metric="euclidean"))

        frame_diff = (frames[:, None] - frames) * -1

        mask = (distances < self.connectivity_radius) & (
            (frame_diff <= self.max_frame_distance) & (frame_diff > 0)
        )

        edge_index = np.argwhere(mask)
        edge_attr = distances[mask]

        return edge_index, edge_attr

    # def compute_ground_truth(self, indices, edge_index, relation):
    #     sender = indices[edge_index[:, 0]]
    #     receiver = indices[edge_index[:, 1]]
    #     self_connections_mask = sender == receiver

    #     relation_indices = relation[:, [-1, 0]]
    #     relation_indices = relation_indices[relation_indices[:, 0] != 0]

    #     relation_mask = np.zeros(len(edge_index), dtype=bool)
    #     for i, (s, r) in enumerate(zip(sender, receiver)):
    #         if np.any((relation_indices == [s, r]).all(1)):
    #             relation_mask[i] = True

    #     ground_truth = self_connections_mask | relation_mask

    #     return ground_truth


def build_graph(segmentations, radius=0.2):
    """
    Convert a list of segmentations into a graph dataset.

    Parameters
    ----------
    segmentations : List[np.ndarray]
        List of segmentations.
    relation : np.ndarray
        Parent-child relation between cells.

    Returns
    -------
    torch_geometric.data.Data
        Graph dataset.
    """
    return GraphFromSegmentations(radius, 2)(segmentations)
