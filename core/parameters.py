from dataclasses import dataclass
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class SystemParameters:
    box_size: float = 600.0
    radii: Dict[str, float] = None
    pair_distances: Dict[str, float] = None
    component_counts: Dict[str, int] = None
    ideal_coordinates: Dict[str, np.ndarray] = None

    def __post_init__(self):
        if self.radii is None:
            self.radii = {'A': 24.0, 'B': 14.0, 'C': 16.0}
        if self.pair_distances is None:
            self.pair_distances = {
                'AA': 48.22,
                'AB': 38.5,
                'BC': 34.0  # minimum allowed C-C distance
            }
        if self.component_counts is None:
            self.component_counts = {'A': 8, 'B': 8, 'C': 16}
        if self.ideal_coordinates is None:
            self.ideal_coordinates = self.latest_ideal()

    def latest_ideal(self) -> Dict[str, np.ndarray]:
        array_A  = np.array([
            [ 63.  ,   0.  ,   0.  ],
            [ 44.55,  44.55,   0.  ],
            [  0.  ,  63.  ,   0.  ],
            [-44.55,  44.55,   0.  ],
            [-63.  ,   0.  ,   0.  ],
            [-44.55, -44.55,   0.  ],
            [ -0.  , -63.  ,   0.  ],
            [ 44.55, -44.55,   0.  ]
            ])
        array_B = np.array([
            [ 63.  ,   0.  , -38.5 ],
            [ 44.55,  44.55, -38.5 ],
            [  0.  ,  63.  , -38.5 ],
            [-44.55,  44.55, -38.5 ],
            [-63.  ,   0.  , -38.5 ],
            [-44.55, -44.55, -38.5 ],
            [ -0.  , -63.  , -38.5 ],
            [ 44.55, -44.55, -38.5 ]
            ]) 
        array_C = np.array([
            [ 47.00,   0.00, -68.50],
            [ 79.00,   0.00, -68.50],
            [ 55.86,  55.86, -68.50],
            [ 33.23,  33.23, -68.50],
            [  0.00,  47.00, -68.50],
            [  0.00,  79.00, -68.50],
            [-55.86,  55.86, -68.50],
            [-33.23,  33.23, -68.50],
            [-47.00,   0.00, -68.50],
            [-79.00,   0.00, -68.50],
            [-55.86, -55.86, -68.50],
            [-33.23, -33.23, -68.50],
            [  0.00, -47.00, -68.50],
            [  0.00, -79.00, -68.50],
            [ 55.86, -55.86, -68.50],
            [ 33.23, -33.23, -68.50],
            ])
        
        return {'A': array_A, 'B': array_B, 'C': array_C}
