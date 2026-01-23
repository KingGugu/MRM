# -*- coding: utf-8 -*-
import random
import numpy as np
from utils import orientation
from scipy.spatial.transform import Rotation as R


class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(self):
        self.data_augmentation_methods = [Sample_substructure(), GaussianCoordinateNoise(), TorsionAnglePerturbation(),
                                          RandomAminoAcidReplacement(), SubspaceCropping(), AminoAcidMasking()]

    def __call__(self, pos, ori, amino_ids):
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        return augment_method(pos, ori, amino_ids)


class Sample_substructure(object):

    def __init__(self):
        self.sample_pct_range = (0.6, 0.8)
        self.random_state = np.random.RandomState(0)

    def __call__(self, pos, ori, amino_ids):

        num_nodes = pos.shape[0]
        if num_nodes == 0:
            return pos, ori, amino_ids, np.expand_dims(np.arange(num_nodes), 1)

        start_idx = self.random_state.randint(0, num_nodes)

        target_pct = self.random_state.uniform(*self.sample_pct_range)
        target_num = max(1, int(num_nodes * target_pct))

        end_idx = start_idx + target_num - 1
        if end_idx >= num_nodes:
            end_idx = num_nodes - 1
            start_idx = max(0, end_idx - target_num + 1)

        sampled_pos = pos[start_idx:end_idx + 1]
        sampled_ori = ori[start_idx:end_idx + 1]
        sampled_amino = amino_ids[start_idx:end_idx + 1]
        sampled_seq = np.expand_dims(np.arange(end_idx - start_idx + 1), 1)

        return sampled_pos, sampled_ori, sampled_amino, sampled_seq


class GaussianCoordinateNoise(object):

    def __init__(self):
        self.sigma_range = (0.05, 0.1)
        self.random_state = np.random.RandomState(0)

    def __call__(self, pos, ori, amino_ids):
        num_nodes = pos.shape[0]
        if num_nodes == 0:
            return pos, ori, amino_ids, np.expand_dims(np.arange(num_nodes), 1)

        sigma = self.random_state.uniform(*self.sigma_range)
        noise = self.random_state.normal(0, sigma, pos.shape)
        noisy_pos = pos + noise

        return (
            noisy_pos,
            ori,
            amino_ids,
            np.expand_dims(np.arange(num_nodes), 1)
        )


class TorsionAnglePerturbation(object):

    def __init__(self):
        self.angle_range = (-5, 5)
        self.random_state = np.random.RandomState(0)

    def __call__(self, pos, ori, amino_ids):
        num_nodes = pos.shape[0]
        if num_nodes < 4:
            return pos, ori, amino_ids, np.expand_dims(np.arange(num_nodes), 1)

        perturbed_pos = pos.copy()
        for i in range(3, num_nodes):
            delta = self.random_state.uniform(*self.angle_range) * np.pi / 180
            p0, p1, p2, p3 = perturbed_pos[i - 3], perturbed_pos[i - 2], perturbed_pos[i - 1], perturbed_pos[i]
            axis = p2 - p1
            axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-6 else np.array([1, 0, 0])
            cos_d, sin_d = np.cos(delta), np.sin(delta)
            cross = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rotation = np.eye(3) + sin_d * cross + (1 - cos_d) * (np.outer(axis, axis) - np.eye(3))

            perturbed_pos[i:] = (rotation @ (perturbed_pos[i:] - p1).T).T + p1

        perturbed_ori = orientation(perturbed_pos)

        return (
            perturbed_pos,
            perturbed_ori,
            amino_ids,
            np.expand_dims(np.arange(num_nodes), 1)
        )


class RandomRotation(object):
    def __init__(self):
        self.random_state = np.random.RandomState(0)

    def __call__(self, pos, ori, amino_ids):
        num_nodes = pos.shape[0]
        if num_nodes == 0:
            return pos, ori, amino_ids, np.expand_dims(np.arange(num_nodes), 1)

        angles = self.random_state.uniform(0, 2 * np.pi, 3)
        rotation = R.from_euler('xyz', angles).as_matrix()
        rotated_pos = pos @ rotation.T
        rotated_ori = ori @ rotation.T

        return (
            rotated_pos,
            rotated_ori,
            amino_ids,
            np.expand_dims(np.arange(num_nodes), 1)
        )


class RandomTranslation(object):
    def __init__(self, max_offset=0.5):
        self.max_offset = max_offset
        self.random_state = np.random.RandomState(0)

    def __call__(self, pos, ori, amino_ids):
        num_nodes = pos.shape[0]
        if num_nodes == 0:
            return pos, ori, amino_ids, np.expand_dims(np.arange(num_nodes), 1)

        offset = self.random_state.uniform(-self.max_offset, self.max_offset, 3)
        translated_pos = pos + offset

        return (
            translated_pos,
            ori,
            amino_ids,
            np.expand_dims(np.arange(num_nodes), 1)
        )


class RandomScaling(object):
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range
        self.random_state = np.random.RandomState(0)

    def __call__(self, pos, ori, amino_ids):
        num_nodes = pos.shape[0]
        if num_nodes == 0:
            return pos, ori, amino_ids, np.expand_dims(np.arange(num_nodes), 1)

        scale = self.random_state.uniform(*self.scale_range)
        scaled_pos = pos * scale

        return (
            scaled_pos,
            ori,
            amino_ids,
            np.expand_dims(np.arange(num_nodes), 1)
        )


class RandomAminoAcidReplacement(object):

    def __init__(self):

        self.aa_list = "ACDEFGHIKLMNPQRSTVWYX"
        self.aa_id = {aa: idx for idx, aa in enumerate(self.aa_list)}

        self.similar_groups = [
            [self.aa_id['A'], self.aa_id['V'], self.aa_id['L'], self.aa_id['I'], self.aa_id['M']],
            [self.aa_id['F'], self.aa_id['Y'], self.aa_id['W']],
            [self.aa_id['S'], self.aa_id['T'], self.aa_id['N'], self.aa_id['Q']],
            [self.aa_id['D'], self.aa_id['E']],
            [self.aa_id['K'], self.aa_id['R'], self.aa_id['H']],
            [self.aa_id['G'], self.aa_id['C'], self.aa_id['P']]
        ]

        self.alanine_id = self.aa_id['A']
        self.replace_pct_range = (0.05, 0.20)
        self.random_state = np.random.RandomState(0)

    def _get_similar_aa(self, aa_idx):
        for group in self.similar_groups:
            if aa_idx in group:
                similar = [idx for idx in group if idx != aa_idx]
                return similar if similar else [aa_idx]
        return [aa_idx]

    def __call__(self, pos, ori, amino_ids):
        num_nodes = pos.shape[0]
        if num_nodes == 0:
            return pos, ori, amino_ids, np.expand_dims(np.arange(num_nodes), 1)

        new_amino_ids = np.copy(amino_ids)

        replace_pct = self.random_state.uniform(*self.replace_pct_range)
        num_replace = max(1, int(num_nodes * replace_pct))

        replace_indices = self.random_state.choice(num_nodes, size=num_replace, replace=False)

        for idx in replace_indices:
            original_aa = new_amino_ids[idx]

            if self.random_state.random() < 0.7:
                similar_aa = self._get_similar_aa(original_aa)
                new_aa = self.random_state.choice(similar_aa)
            else:
                new_aa = self.alanine_id

            new_amino_ids[idx] = new_aa

        return (
            pos,
            ori,
            new_amino_ids,
            np.expand_dims(np.arange(num_nodes), 1)
        )


class AminoAcidMasking(object):

    def __init__(self, mask_pct_range=(0.05, 0.20), mask_id=20):
        self.mask_pct_range = mask_pct_range
        self.mask_id = mask_id
        self.random_state = np.random.RandomState(0)

    def __call__(self, pos, ori, amino_ids):
        num_nodes = pos.shape[0]
        if num_nodes == 0:
            return pos, ori, amino_ids, np.expand_dims(np.arange(num_nodes), 1)

        masked_amino_ids = np.copy(amino_ids)

        mask_pct = self.random_state.uniform(*self.mask_pct_range)
        num_mask = max(1, int(num_nodes * mask_pct))
        mask_indices = self.random_state.choice(num_nodes, size=num_mask, replace=False)
        masked_amino_ids[mask_indices] = self.mask_id

        return (
            pos,
            ori,
            masked_amino_ids,
            np.expand_dims(np.arange(num_nodes), 1)
        )


class SubspaceCropping(object):

    def __init__(self,
                 radius_range=(10.0, 20.0),
                 min_residues=5,
                 max_residues=200,
                 center_prob_strategy="uniform",
                 random_state=0):

        self.radius_range = radius_range
        self.min_residues = min_residues
        self.max_residues = max_residues
        self.center_prob_strategy = center_prob_strategy
        self.random_state = np.random.RandomState(random_state)

    def _select_center(self, pos, amino_ids):
        num_nodes = pos.shape[0]
        if self.center_prob_strategy == "uniform":
            return self.random_state.choice(num_nodes)
        else:
            return self.random_state.choice(num_nodes)

    def _get_spatial_neighbors(self, pos, center_idx, radius):
        center_pos = pos[center_idx]
        distances = np.linalg.norm(pos - center_pos, axis=1)
        return np.where(distances <= radius)[0]

    def __call__(self, pos, ori, amino_ids):
        num_nodes = pos.shape[0]
        if num_nodes == 0:
            return pos, ori, amino_ids, np.expand_dims(np.arange(num_nodes), 1)

        if num_nodes <= self.min_residues:
            return (
                pos,
                ori,
                amino_ids,
                np.expand_dims(np.arange(num_nodes), 1)
            )

        center_idx = self._select_center(pos, amino_ids)

        crop_radius = self.random_state.uniform(*self.radius_range)

        selected_indices = self._get_spatial_neighbors(pos, center_idx, crop_radius)
        selected_count = len(selected_indices)

        if selected_count < self.min_residues:
            selected_indices = self._get_spatial_neighbors(pos, center_idx, self.radius_range[1])
            selected_count = len(selected_indices)
            if selected_count < self.min_residues:
                return (
                    pos,
                    ori,
                    amino_ids,
                    np.expand_dims(np.arange(num_nodes), 1)
                )
        elif selected_count > self.max_residues:
            center_pos = pos[center_idx]
            distances = np.linalg.norm(pos[selected_indices] - center_pos, axis=1)
            sorted_indices = np.argsort(distances)[:self.max_residues]
            selected_indices = selected_indices[sorted_indices]

        selected_indices = np.sort(selected_indices)

        cropped_pos = pos[selected_indices]
        cropped_ori = ori[selected_indices]
        cropped_amino = amino_ids[selected_indices]
        cropped_seq = np.expand_dims(np.arange(len(selected_indices)), 1)

        return (
            cropped_pos,
            cropped_ori,
            cropped_amino,
            cropped_seq
        )
