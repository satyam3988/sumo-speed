"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        # self.ts = ts
        super().__init__(ts)

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        vehs = self._get_veh_list()
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        # average_speeds = self.ts.get_lanes_average_speed()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32) # old
        # observation = np.array(phase_id + min_green + density + queue + average_speeds, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32), # old
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32), # old
            # low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes) + len(self.ts.lanes), dtype=np.float32), # each element is for each observation space
            # high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes) + len(self.ts.lanes), dtype=np.float32),
        )

    def function_name(self) -> str:
        """Return the name of the observation function."""
        return "default"