# from .matrix_lie_group import MatrixLieGroupBase
from kornia.core import Tensor, concatenate, stack
from .quaternion import Quaternion
from kornia.testing import KORNIA_CHECK, KORNIA_CHECK_SHAPE, KORNIA_CHECK_TYPE
from torch import linalg as LA
from __future__ import annotations
# from overrides import overrides


# class SO3Matrix():
#     """See :mod:`liegroups.SO3`"""
#     dim = 3
#     dof = 3

class SO3():
    """Special orthogonal group for 3D rotations.

    Internal parameterization is `(qw, qx, qy, qz)`. 
    Tangent parameterization is `(omega_x, omega_y, omega_z)`.
    """

    matrix_dim: int = 3
    parameters_dim: int = 4
    tangent_dim: int = 3
    space_dim: int = 3

    def __init__(self, data: Quaternion) -> None:

        KORNIA_CHECK_TYPE(data, Quaternion)
        KORNIA_CHECK_SHAPE(data, ["B", "4"])

        assert len(data.shape) == 2 and data.shape[-1] == 4
        if not (len(data.shape) == 2 and data.shape[-3] == 4):
            raise ValueError(f"Input data musth have a shape (B, 4). Got: {data.shape}")
        self.data = data

    def __repr__(self) -> str:
        data = Tensor.round(self.data, 5)
        return f"{self.__class__.__name__}(data={data})"

    def __getattr__(self, name: str):
        """Direct access to torch methods."""
        return getattr(self.data, name)

    def __getitem__(self, idx) -> Tensor:
        return self.data[idx]

    @classmethod
    def from_x_radians(cls, theta: Tensor) -> SO3:
        """Generates a x-axis rotation.

        Args:
            angle: X rotation, in radians.

        Returns:
            Output.
        """
        return cls.exp(Tensor([theta, 0.0, 0.0]))

    @classmethod
    def from_y_radians(cls, theta: Tensor) -> SO3:
        """Generates a y-axis rotation.

        Args:
            angle: Y rotation, in radians.

        Returns:
            Output.
        """
        return cls.exp(Tensor([0.0, theta, 0.0]))

    @classmethod
    def from_z_radians(cls, theta: Tensor) -> SO3:
        """Generates a z-axis rotation.

        Args:
            angle: Z rotation, in radians.

        Returns:
            Output.
        """
        return cls.exp(Tensor([0.0, 0.0, theta]))

    @classmethod
    def from_rpy_radians(cls, roll: Tesnor, pitch: Tesnor, yaw: Tesnor) -> SO3:
        """Generates a transform from a set of Euler angles. Uses the ZYX mobile robot
        convention.

        Args:
            roll: X rotation, in radians. Applied first.
            pitch: Y rotation, in radians. Applied second.
            yaw: Z rotation, in radians. Applied last.

        Returns:
            Output.
        """
        return (
            cls.from_z_radians(yaw)
            @ cls.from_y_radians(pitch)
            @ cls.from_x_radians(roll)
        )

    def compute_roll_radians(self) -> Tensor:
        """Compute roll angle. Uses the ZYX mobile robot convention.

        Returns:
            Euler angle in radians.
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        q0, q1, q2, q3 = self.data
        return torch.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))

    def compute_pitch_radians(self) -> Tensor:
        """Compute pitch angle. Uses the ZYX mobile robot convention.

        Returns:
            Euler angle in radians.
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        q0, q1, q2, q3 = self.data
        return torch.arcsin(2 * (q0 * q2 - q3 * q1))

    def compute_yaw_radians(self) -> Tensor:
        """Compute yaw angle. Uses the ZYX mobile robot convention.

        Returns:
            Euler angle in radians.
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        q0, q1, q2, q3 = self.data
        return torch.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

    def as_rpy_radians(self) -> Tesnor:
        """Computes roll, pitch, and yaw angles. Uses the ZYX mobile robot convention.

        Returns:
            Named tuple containing Euler angles in radians.
        """
        return Tesnor(
            roll=self.compute_roll_radians(),
            pitch=self.compute_pitch_radians(),
            yaw=self.compute_yaw_radians(),
        )

    @classmethod
    def identity(cls) -> SO3:
        return cls(data=Tensor([1.0, 0.0, 0.0, 0.0]))

    def matrix(self) -> Tensor:
        return self.data.matrix()

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> SO3:
        return cls(data=Quaternion.from_matrix(matrix))

    @classmethod
    def exp(cls, tangent: Tensor) -> 'SO3':
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/so3.hpp#L583

        KORNIA_CHECK_SHAPE(tangent, ["B", "3"])

        theta_sq = tangent.pow(2).sum(-1, True)
        theta_po4 = theta_sq.pow(2)
        use_taylor = theta_sq < torch.finfo(tangent.dtype).eps

        theta = torch.sqrt(theta_sq)
        half_theta = 0.5 * theta

        real_factor = torch.where(
            use_taylor,
            # real_factor = Scalar(1) - Scalar(1.0 / 8.0) * theta_sq +
            #         Scalar(1.0 / 384.0) * theta_po4;
            1.0 - theta_sq / 8.0 + theta_po4 / 384.0,
            # real_factor = cos(half_theta);
            torch.cos(half_theta),
        )

        imaginary_factor = torch.where(
            use_taylor,
            # imag_factor = Scalar(0.5) - Scalar(1.0 / 48.0) * theta_sq +
            #         Scalar(1.0 / 3840.0) * theta_po4;
            0.5 - theta_sq / 48.0 + theta_po4 / 3840.0,
            # imag_factor = sin_half_theta / (*theta);
            torch.sin(half_theta) / theta,get.
        )

        print(real_factor)
        print(imaginary_factor * tangent)
        print(concatenate([real_factor, imaginary_factor * tangent], dim=-1).shape)

        return cls(data=Quaternion(concatenate([real_factor, imaginary_factor * tangent], dim=-1)))

    def log(self) -> Tensor:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/so3.hpp#L247

        w = self.data.w()
        vec = self.data.vec()
        squared_n = vec.pow(2).sum(-1, True)
        use_taylor = squared_n < torch.finfo(self.w.dtype).eps

        norm_safe = torch.sqrt(squared_n)

        atan_n_over_w = torch.arctan2(
            torch.where(w < 0, -norm_safe, norm_safe),
            torch.abs(w),
        )

        atan_factor = torch.where(
            use_taylor,
            2.0 / w - 2.0 / 3.0 * squared_n / w**3,
            torch.where(
                torch.abs(w) < torch.finfo(self.w.dtype).eps,
                torch.where(w > 0, 1.0, -1.0) * torch.pi / norm_safe,
                2.0 * atan_n_over_w / norm_safe,
            ),
        )

        return atan_factor * vec

    def adjoint(self) -> Tensor:
        return self.as_matrix()

    def parameters(self) -> Tensor:
        return self.data

    def multiply(self, other: SO3) -> SO3:
        return SO3(data=self.data * other.data)

    def inverse(self) -> SO3:
        # Negate complex terms.
        return SO3(data=self.data.inv())

    def normalize(self) -> SO3:
        return SO3(data=self.data.normalize())
