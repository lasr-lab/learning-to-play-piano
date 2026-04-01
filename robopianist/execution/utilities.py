from typing import Optional

from dm_env.specs import BoundedArray


class KindBoundedArray(BoundedArray):
    def __init__(self, shape, dtype, minimum, maximum, tolerance=None, name: Optional[str] = None):
        """
        A BoundedArray spec that allows for a tolerance when validating values.
        """
        super().__init__(shape, dtype, minimum, maximum, name)
        if tolerance is None:
            self._testing_spec = super()
        else:
            self._testing_spec = BoundedArray(shape, dtype, minimum - tolerance, maximum + tolerance, name)

    def validate(self, value):
        self._testing_spec.validate(value)
