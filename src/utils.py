import os
import random
from typing import Tuple, Optional
import numpy as np
import torch
from transforms3d.quaternions import quat2mat

from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from .type import Grasp
from .constants import PC_MAX, PC_MIN


def to_pose(
    trans: Optional[np.ndarray] = None, rot: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert translation and rotation into a 4x4 pose matrix.

    Parameters
    ----------
    trans: Optional[np.ndarray]
        Translation vector, shape (3,).
    rot: Optional[np.ndarray]
        Rotation matrix, shape (3, 3).

    Returns
    -------
    np.ndarray
        4x4 pose matrix.
    """
    ret = np.eye(4)
    if trans is not None:
        ret[:3, 3] = trans
    if rot is not None:
        ret[:3, :3] = rot
    return ret


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed: int
        Random seed between 0 and 2**32 - 1
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def transform_grasp_pose(
    grasp: Grasp,
    est_trans: np.ndarray,
    est_rot: np.ndarray,
    cam_trans: np.ndarray,
    cam_rot: np.ndarray,
) -> Grasp:
    """
    Transform grasp from the object frame into the robot frame

    Parameters
    ----------
    grasp: Grasp
        The grasp to be transformed.
    est_trans: np.ndarray
        Estimated translation vector in the camera frame.
    est_rot: np.ndarray
        Estimated rotation matrix in the camera frame.
    cam_trans: np.ndarray
        Camera's translation vector in the robot frame.
    cam_rot: np.ndarray
        Camera's rotation matrix in the robot frame.

    Returns
    -------
    Grasp
        The transformed grasp in the robot frame.
    """
    # raise NotImplementedError

    trans_in_cam = est_rot @ grasp.trans + est_trans
    rot_in_cam = est_rot @ grasp.rot
    trans_in_robot = cam_rot @ trans_in_cam + cam_trans
    rot_in_robot = cam_rot @ rot_in_cam

    return Grasp(trans_in_robot, rot_in_robot, grasp.width)



def get_pc(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Convert depth image into point cloud using intrinsics

    All points with depth=0 are filtered out

    Parameters
    ----------
    depth: np.ndarray
        Depth image, shape (H, W)
    intrinsics: np.ndarray
        Intrinsics matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        Point cloud with shape (N, 3)
    """
    # Get image dimensions
    height, width = depth.shape
    # Create meshgrid for pixel coordinates
    v, u = np.meshgrid(range(height), range(width), indexing="ij")
    # Flatten the arrays
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth.flatten()
    # Filter out invalid depth values
    valid = depth_flat > 0
    u = u[valid]
    v = v[valid]
    depth_flat = depth_flat[valid]
    # Create homogeneous pixel coordinates
    pixels = np.stack([u, v, np.ones_like(u)], axis=0)
    # Convert pixel coordinates to camera coordinates
    rays = np.linalg.inv(intrinsics) @ pixels
    # Scale rays by depth
    points = rays * depth_flat
    return points.T


def get_workspace_mask(pc: np.ndarray) -> np.ndarray:
    """Get the mask of the point cloud in the workspace."""
    pc_mask = (
        (pc[:, 0] > PC_MIN[0])
        & (pc[:, 0] < PC_MAX[0])
        & (pc[:, 1] > PC_MIN[1])
        & (pc[:, 1] < PC_MAX[1])
        & (pc[:, 2] > PC_MIN[2])
        & (pc[:, 2] < PC_MAX[2])
    )
    return pc_mask


def rand_rot_mat() -> np.ndarray:
    """
    Generate a random rotation matrix with shape (3, 3) uniformly.
    """
    while True:
        quat = np.random.randn(4)
        if np.linalg.norm(quat) > 1e-6:
            break
    quat /= np.linalg.norm(quat)
    return quat2mat(quat)


def theta_to_2d_rot(theta: float) -> np.ndarray:
    """
    Convert a 2D rotation angle into a rotation matrix.

    Parameters
    ----------
    theta : float
        The rotation angle in radians.

    Returns
    -------
    np.ndarray
        The resulting 2D rotation matrix (2, 2).
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rot_dist(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    The relative rotation angle between two rotation matrices.

    Parameters
    ----------
    r1 : np.ndarray
        The first rotation matrix (3, 3).
    r2 : np.ndarray
        The second rotation matrix (3, 3).

    Returns
    -------
    float
        The relative rotation angle in radians.
    """
    return np.arccos(np.clip((np.trace(r1 @ r2.T) - 1) / 2, -1, 1))


# You can add additional functions here

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss