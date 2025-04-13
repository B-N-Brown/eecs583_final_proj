import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from compiler_gym.spaces import Space
from compiler_gym.service import observation_pb2


class LoopObservationSpace(Space):
    """An observation space that provides information about loops in the program.
    
    This space returns information about loops including:
    - Number of loops
    - Loop depth
    - Loop iteration counts (when available)
    - Loop body size
    - Loop entry and exit blocks
    """
    
    @todo
    def __init__(self, service):
        """Initialize a new loop observation space.
        
        Args:
            service: The CompilerGym service that provides the loop information.
        """