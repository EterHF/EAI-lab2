from typing import Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F

from ..config import Config
from src.utils import PointNetEncoder, feature_transform_reguliarzer


class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config
        # raise NotImplementedError("You need to implement some modules here")
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.rot_head = nn.Linear(256, 9)
        self.trans_head = nn.Linear(256, 3)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        # raise NotImplementedError("You need to implement the forward function")

        pc = pc.transpose(1, 2)
        pc, _trans, _trans_feat = self.feat(pc)
        pc = F.relu(self.bn1(self.fc1(pc)))
        pc = F.relu(self.bn2(self.dropout(self.fc2(pc))))

        est_rot = self.rot_head(pc)
        est_trans = self.trans_head(pc)
        
        # vec1 = est_rot[:, 0:3] / torch.norm(est_rot[:, 0:3], dim=1, keepdim=True)
        # vec2 = est_rot[:, 3:6]
        # proj = torch.sum(vec2 * vec1, dim=1, keepdim=True) * vec1
        # vec2 = vec2 - proj
        # vec2 = vec2 / torch.norm(vec2, dim=1, keepdim=True)
        # vec3 = torch.cross(vec1, vec2, dim=1)
        # est_rot = torch.stack([vec1, vec2, vec3], dim=1)
        est_rot = est_rot.view(-1, 3, 3)
        est_trans = est_trans.view(-1, 3)

        # SVD
        U, S, Vh = torch.linalg.svd(est_rot, full_matrices=False)  # U: (..., 3, 3), Vh: (..., 3, 3)
        V = Vh.mH
        est_rot = U @ V
        # det(R) = 1
        det = torch.linalg.det(est_rot)
        sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)
        V_corrected = V * sign
        est_rot = U @ V_corrected.mH

        rot_loss = F.mse_loss(est_rot, rot)
        rot_diff = torch.bmm(est_rot, rot.transpose(1, 2))
        trace = torch.einsum('bii->b', rot_diff) 
        angle_loss = torch.arccos(torch.clamp((trace - 1) / 2, -0.99999, 0.99999)).mean()
        trans_loss = F.smooth_l1_loss(est_trans, trans)
        
        loss = rot_loss + angle_loss + 0.1 * trans_loss + 0.001 * feature_transform_reguliarzer(_trans_feat)
        metric = dict(
            loss=loss,
            trans_loss=trans_loss,
            rot_loss=rot_loss,
            angle_loss=angle_loss, 
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
        """
        # raise NotImplementedError("You need to implement the est function")

        with torch.no_grad():
            pc = pc.transpose(1, 2)
            pc, _trans, _trans_feat = self.feat(pc)
            pc = F.relu(self.bn1(self.fc1(pc)))
            pc = F.relu(self.bn2(self.dropout(self.fc2(pc))))
            est_rot = self.rot_head(pc)
            est_trans = self.trans_head(pc)

            est_rot = est_rot.view(-1, 3, 3)
            est_trans = est_trans.view(-1, 3)

            # SVD
            U, S, Vh = torch.linalg.svd(est_rot, full_matrices=False)  # U: (..., 3, 3), Vh: (..., 3, 3)
            V = Vh.mH
            est_rot = U @ V
            # det(R) = 1
            det = torch.linalg.det(est_rot)
            sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)
            V_corrected = V * sign
            est_rot = U @ V_corrected.mH
        
        return est_trans, est_rot
