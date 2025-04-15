from typing import Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F

from ..config import Config
from src.utils import STN3d, STNkd


class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config
        # raise NotImplementedError("You need to implement some modules here")

        self.part_num = 4
        self.stn = STN3d(3)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fstn = STNkd(k=128)

        self.fc_trans = nn.Linear(1024, 512)
        self.fc_trans2 = nn.Linear(512, 256)
        self.fc_trans3 = nn.Linear(256, 3)
        
        self.fc_rot = nn.Linear(1024, 512)
        self.fc_rot2 = nn.Linear(512, 256)
        self.fc_rot3 = nn.Linear(256, 9)
        
        self.bn_trans1 = nn.BatchNorm1d(512)
        self.bn_trans2 = nn.BatchNorm1d(256)
        
        self.bn_rot1 = nn.BatchNorm1d(512)
        self.bn_rot2 = nn.BatchNorm1d(256)
        
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

        pc = pc.transpose(2, 1)
        B, _, N = pc.size()
        _trans = self.stn(pc)
        pc = pc.transpose(2, 1)
        pc = torch.bmm(pc, _trans)
        pc = pc.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(pc)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 1024)

        est_trans = F.relu(self.bn_trans1(self.fc_trans(out_max)))
        est_trans = F.relu(self.bn_trans2(self.fc_trans2(est_trans)))
        est_trans = self.fc_trans3(est_trans)
        est_rot = F.relu(self.bn_rot1(self.fc_rot(out_max)))
        est_rot = F.relu(self.bn_rot2(self.fc_rot2(est_rot)))
        est_rot = self.fc_rot3(est_rot)
        
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

        rot_loss = F.mse_loss(est_rot - rot, dim=(1, 2))
        trans_loss = F.l1_loss(est_trans - trans)
        
        loss = rot_loss + trans_loss
        metric = dict(
            loss=loss,
            trans_loss=trans_loss,
            rot_loss=rot_loss
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
            pc = pc.transpose(2, 1)
            B, _, N = pc.size()
            _trans = self.stn(pc)
            pc = pc.transpose(2, 1)
            pc = torch.bmm(pc, _trans)
            pc = pc.transpose(2, 1)

            out1 = F.relu(self.bn1(self.conv1(pc)))
            out2 = F.relu(self.bn2(self.conv2(out1)))
            out3 = F.relu(self.bn3(self.conv3(out2)))

            trans_feat = self.fstn(out3)
            x = out3.transpose(2, 1)
            net_transformed = torch.bmm(x, trans_feat)
            net_transformed = net_transformed.transpose(2, 1)

            out4 = F.relu(self.bn4(self.conv4(net_transformed)))
            out5 = self.bn5(self.conv5(out4))
            out_max = torch.max(out5, 2, keepdim=True)[0]
            out_max = out_max.view(-1, 1024)

            est_trans = F.relu(self.bn_trans1(self.fc_trans(out_max)))
            est_trans = F.relu(self.bn_trans2(self.fc_trans2(est_trans)))
            est_trans = self.fc_trans3(est_trans)
            est_rot = F.relu(self.bn_rot1(self.fc_rot(out_max)))
            est_rot = F.relu(self.bn_rot2(self.fc_rot2(est_rot)))
            est_rot = self.fc_rot3(est_rot)
            
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
        
        return est_trans, est_rot
