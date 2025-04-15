from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..config import Config
from ..vis import Vis
from src.utils import PointNetEncoder


class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config
        # raise NotImplementedError("You need to implement some modules here")
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=3)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 3, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        # raise NotImplementedError("You need to implement the forward function")

        pc = pc.transpose(1, 2)
        batchsize = pc.size()[0]
        n_pts = pc.size()[2]
        pc, _, _ = self.feat(pc)
        pc = F.relu(self.bn1(self.conv1(pc)))
        pc = F.relu(self.bn2(self.conv2(pc)))
        pc = F.relu(self.bn3(self.conv3(pc)))
        pc = self.conv4(pc)
        pc = pc.transpose(2,1).contiguous()
        # pc = F.log_softmax(pc.view(-1, 3), dim=-1)
        est_coord = pc.view(batchsize, n_pts, 3)

        loss = F.mse_loss(est_coord, coord)
        metric = dict(
            loss=loss,
            # additional metrics you want to log
        )
        return loss, metric

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        # raise NotImplementedError("You need to implement the est function")

        pc = pc.transpose(1, 2)
        batchsize = pc.size()[0]
        n_pts = pc.size()[2]
        pc, _, _ = self.feat(pc)
        pc = F.relu(self.bn1(self.conv1(pc)))
        pc = F.relu(self.bn2(self.conv2(pc)))
        pc = F.relu(self.bn3(self.conv3(pc)))
        pc = self.conv4(pc)
        pc = pc.transpose(2,1).contiguous()
        # pc = F.log_softmax(pc.view(-1, 3), dim=-1)
        est_coord = pc.view(batchsize, n_pts, 3)

        device = pc.device
        dtype = pc.dtype
        batch_trans = []
        batch_rot = []
        
        for b in range(batchsize):
            src_points = pc[b].detach().cpu().numpy()
            dst_points = est_coord[b].detach().cpu().numpy()
            
            # parameters of RANSAC
            best_inliers = 0
            best_R = np.eye(3)
            best_t = np.zeros(3)
            iterations = 1000
            threshold = 0.0001
            min_samples = 3
            
            for _ in range(iterations):
                idx = np.random.choice(n_pts, min_samples, replace=False)
                p1 = src_points[idx]
                p2 = dst_points[idx]
                
                H = p2.T @ p1
                U, _, Vt = np.linalg.svd(H)
                R = U @ Vt
                
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = U @ Vt
                
                t = p2[0] - R @ p1[0]
                errors = np.linalg.norm(dst_points - (src_points @ R.T + t), axis=1)
                inliers = np.sum(errors < threshold)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_R = R
                    best_t = t
            
            errors = np.linalg.norm(dst_points - (src_points @ best_R.T + best_t), axis=1)
            inlier_mask = errors < threshold
            
            if np.sum(inlier_mask) >= min_samples:
                p1 = src_points[inlier_mask]
                p2 = dst_points[inlier_mask]
                
                H = p2.T @ p1
                U, _, Vt = np.linalg.svd(H)
                R = U @ Vt

                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = U @ Vt
                
                t = p2[0] - R @ p1[0]
                best_R = R
                best_t = t
            
            batch_rot.append(torch.tensor(best_R, device=device, dtype=dtype))
            batch_trans.append(torch.tensor(best_t, device=device, dtype=dtype))
        
        rot = torch.stack(batch_rot)
        trans = torch.stack(batch_trans)

        return trans, rot