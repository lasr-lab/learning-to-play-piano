import abc
from pathlib import Path

from dm_control import composer, mjcf
from mujoco_utils import mjcf_utils, types

class Cube(composer.Entity, abc.ABC):
    def _build(self) -> None:
        _HERE = Path(__file__).resolve().parent
        cube_path = _HERE / "third_party" / "cube" / "my_cube.xml"
        self._mjcf_root = mjcf.from_path(cube_path)
        self._mjcf_root.model = "cube"

    @property
    def mjcf_model(self) -> types.MjcfRootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._mjcf_root.model