"""
Sigma Prior Management
======================
Centralized handling of sigma (nuisance parameter) initialization and priors.
"""
import os
import json
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from scipy import stats

#==================================================================================
# GLOBAL CONFIGURATION: Default sigma ranges for all pair types
#==================================================================================
DEFAULT_SIGMA_RANGES = {
    'AA': (0.24, 7.0),
    'AB': (0.14, 4.5),
    'BC': (0.16, 5.0)
}

#==================================================================================
# Default hyperparameters for parametric priors - BETTER CALIBRATION
#==================================================================================
DEFAULT_GAMMA_PARAMS = {
    # For AA pairs (larger particles, ~48nm diameter, expect sigma ~ 1-2nm)
    'AA': {'shape': 6.0, 'scale': 0.4},   # mean=1.6, std=0.8, 95% within (0.4, 3.3)
    
    # For AB pairs (mixed sizes, expect sigma ~ 0.5-1.5nm)
    'AB': {'shape': 6.0, 'scale': 0.25},  # mean=1.0, std=0.5, 95% within (0.25, 2.1)
    
    # For BC pairs (small particles, expect sigma ~ 0.8-2nm)
    'BC': {'shape': 6.0, 'scale': 0.35}   # mean=1.4, std=0.7, 95% within (0.35, 2.9)
}


DEFAULT_HALF_CAUCHY_PARAMS = {
    'AA': {'scale': 1.0},   # median = scale
    'AB': {'scale': 0.6},
    'BC': {'scale': 0.8}
}

#==================================================================================
# Prior Distributions Factory
#==================================================================================
def get_prior_distribution(prior_type: str, low: float, high: float, 
                          hyperparams: Optional[Dict[str, float]] = None):
    """Get a prior distribution object based on type and parameters."""
    if prior_type == 'uniform':
        return stats.uniform(loc=low, scale=high-low)
    
    elif prior_type in ['jeffreys', 'log_uniform']:
        return stats.loguniform(a=low, b=high)
    
    elif prior_type == 'gamma':
        if hyperparams is None:
            mean = (low + high) / 2
            cv = 0.5
            shape = 1 / (cv ** 2)
            scale = mean / shape
        else:
            shape = hyperparams.get('shape', 2.0)
            scale = hyperparams.get('scale', 1.5)
        
        return stats.gamma(a=shape, scale=scale)
    
    elif prior_type == 'inv_gamma':
        # Inverse-Gamma is conjugate prior for σ²
        # Parameterized so E[σ] ≈ target, with concentration around it
        if hyperparams is None:
            # Default: concentrated around 1nm with moderate spread
            shape = 5.0   # Higher shape = more concentrated
            scale = 4.0   # E[σ] = sqrt(scale/(shape-1)) for shape > 1
        else:
            shape = hyperparams.get('shape', 5.0)
            scale = hyperparams.get('scale', 4.0)
        
        return stats.invgamma(a=shape, scale=scale)
        
    
    elif prior_type == 'half_cauchy':
        if hyperparams is None:
            scale = (low + high) / 2
        else:
            scale = hyperparams.get('scale', 2.0)
        
        return stats.halfcauchy(scale=scale)
    
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

#==================================================================================
# SigmaPrior Class
#==================================================================================
class SigmaPrior:
    """Prior distribution for sigma parameters."""
    
    def __init__(self, 
                 pair_types: List[str],
                 sigma_ranges: Dict[str, Tuple[float, float]],
                 prior_type: str = "uniform",
                 use_gmm: bool = False,
                 gmm_params: Optional[Dict[str, Dict[str, Any]]] = None,
                 hyperparams: Optional[Dict[str, Dict[str, float]]] = None):
        self.pair_types = pair_types
        self.sigma_ranges = sigma_ranges
        self.prior_type = prior_type
        self.use_gmm = use_gmm
        self.gmm_params = gmm_params or {}
        self.hyperparams = hyperparams or {}
        
        if prior_type == 'gamma' and not hyperparams:
            self.hyperparams = DEFAULT_GAMMA_PARAMS.copy()
        elif prior_type == 'half_cauchy' and not hyperparams:
            self.hyperparams = DEFAULT_HALF_CAUCHY_PARAMS.copy()
        
        for pt in pair_types:
            if pt not in sigma_ranges:
                raise ValueError(f"Missing sigma range for pair type: {pt}")
    
    def initialize_sigma(self, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """Sample initial sigma values from prior - FIXED VERSION"""
        if rng is None:
            rng = np.random.default_rng()
        
        sigma = {}
        
        if self.use_gmm:
            # Sample from GMM
            for pair_type in self.pair_types:
                gmm = self.gmm_params[pair_type]
                weights = np.array(gmm['weights'])
                means = np.array(gmm['means']).flatten()
                stds = np.sqrt(np.array(gmm['covariances']).flatten())
                
                component = rng.choice(len(weights), p=weights/np.sum(weights))
                value = rng.normal(means[component], stds[component])
                
                low, high = self.sigma_ranges[pair_type]
                sigma[pair_type] = float(np.clip(value, low, high))
        
        elif self.prior_type in ['uniform', 'jeffreys', 'log_uniform']:
            # Sample in log-space for better coverage - FIXED!
            for pair_type in self.pair_types:
                low, high = self.sigma_ranges[pair_type]
                log_val = rng.uniform(np.log(low), np.log(high))
                sigma[pair_type] = float(np.exp(log_val))  # REMOVED hardcoded override!
        
        else:  # Parametric priors (gamma, half_cauchy)
            for pair_type in self.pair_types:
                low, high = self.sigma_ranges[pair_type]
                hp = self.hyperparams.get(pair_type, None)
                
                dist = get_prior_distribution(self.prior_type, low, high, hp)
                value = dist.rvs(random_state=rng)
                sigma[pair_type] = float(np.clip(value, low, high))  # REMOVED hardcoded override!
        
        return sigma
    
    def log_prior(self, sigma: Dict[str, float]) -> float:
        """Calculate log prior probability for given sigma values."""
        if self.use_gmm:
            return self._gmm_log_prior(sigma)
        else:
            return self._parametric_log_prior(sigma)
    
    def _parametric_log_prior(self, sigma: Dict[str, float]) -> float:
        """Calculate log prior using parametric distributions"""
        log_prob = 0.0
        
        for pair_type, value in sigma.items():
            low, high = self.sigma_ranges[pair_type]
            
            # Hard boundary check for bounded priors
            if self.prior_type not in ['half_cauchy', 'gamma']:
                if not (low <= value <= high):
                    return -np.inf
            else:
                # For unbounded priors, soft penalty
                if value < low:
                    return -np.inf
                if value > high:
                    log_prob -= 1000 * (value - high)
            
            # Evaluate log probability
            hp = self.hyperparams.get(pair_type, None)
            dist = get_prior_distribution(self.prior_type, low, high, hp)
            log_prob += dist.logpdf(value)
        
        return log_prob
    
    def _gmm_log_prior(self, sigma: Dict[str, float]) -> float:
        """Calculate log prior using GMM from previous sampler."""
        log_prob = 0.0
        
        for pair_type, value in sigma.items():
            low, high = self.sigma_ranges[pair_type]
            if not (low <= value <= high):
                return -np.inf
            
            gmm = self.gmm_params[pair_type]
            weights = np.array(gmm['weights'])
            means = np.array(gmm['means']).flatten()
            covs = np.array(gmm['covariances']).flatten()
            
            covs = np.maximum(covs, 1e-10)
            weights = weights / np.sum(weights)
            
            stds = np.sqrt(covs)
            component_logprobs = (
                np.log(weights) 
                - 0.5 * np.log(2 * np.pi * covs)
                - 0.5 * ((value - means) / stds) ** 2
            )
            
            max_logprob = np.max(component_logprobs)
            log_prob += max_logprob + np.log(np.sum(np.exp(component_logprobs - max_logprob)))
        
        return log_prob
    
    @classmethod
    def from_gmm_file(cls, 
                     gmm_file: str, 
                     prior_type: str = "uniform",
                     sigma_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> 'SigmaPrior':
        """Load SigmaPrior from GMM parameters JSON file."""
        try:
            with open(gmm_file, 'r') as f:
                gmm_params = json.load(f)
            
            pair_types = list(gmm_params.keys())
            
            if sigma_ranges is None:
                sigma_ranges = {}
                for pt in pair_types:
                    means = np.array(gmm_params[pt]['means']).flatten()
                    stds = np.sqrt(np.array(gmm_params[pt]['covariances']).flatten())
                    low = max(0.1, np.min(means - 3*stds))
                    high = np.max(means + 3*stds)
                    sigma_ranges[pt] = (float(low), float(high))
            
            return cls(
                pair_types=pair_types,
                sigma_ranges=sigma_ranges,
                prior_type=prior_type,
                use_gmm=True,
                gmm_params=gmm_params
            )
            
        except Exception as e:
            print(f"Warning: Failed to load GMM from {gmm_file}: {e}")
            print(f"Falling back to simple {prior_type} prior")
            
            if sigma_ranges is None:
                sigma_ranges = DEFAULT_SIGMA_RANGES.copy()
            
            pair_types = list(sigma_ranges.keys())
            
            return cls(
                pair_types=pair_types,
                sigma_ranges=sigma_ranges,
                prior_type=prior_type,
                use_gmm=False
            )

#==================================================================================
# Factory Functions
#==================================================================================
def get_default_sigma_ranges(pair_types: Optional[List[str]] = None) -> Dict[str, Tuple[float, float]]:
    """Get default sigma ranges for specified pair types."""
    if pair_types is None:
        return DEFAULT_SIGMA_RANGES.copy()
    
    return {pt: DEFAULT_SIGMA_RANGES.get(pt, (1.0, 10.0)) for pt in pair_types}

def get_default_hyperparams(prior_type: str, 
                            pair_types: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """Get default hyperparameters for parametric priors."""
    if prior_type == 'gamma':
        base = DEFAULT_GAMMA_PARAMS
    elif prior_type == 'half_cauchy':
        base = DEFAULT_HALF_CAUCHY_PARAMS
    else:
        return {}
    
    if pair_types is None:
        return base.copy()
    
    return {pt: base.get(pt, base['AA']) for pt in pair_types}

def initialize_sigma_dict(pair_types: Optional[List[str]] = None,
                         sigma_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                         rng: Optional[np.random.Generator] = None,
                         prior_type: str = "uniform",
                         hyperparams: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
    """Initialize sigma dictionary with random values."""
    if pair_types is None:
        pair_types = list(DEFAULT_SIGMA_RANGES.keys())
    
    if sigma_ranges is None:
        sigma_ranges = get_default_sigma_ranges(pair_types)
    
    if hyperparams is None and prior_type in ['gamma', 'half_cauchy']:
        hyperparams = get_default_hyperparams(prior_type, pair_types)
    
    prior = SigmaPrior(
        pair_types=pair_types,
        sigma_ranges=sigma_ranges,
        prior_type=prior_type,
        hyperparams=hyperparams
    )
    
    return prior.initialize_sigma(rng)

def create_sigma_prior(pair_types: List[str],
                      sigma_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                      gmm_file: Optional[str] = None,
                      prior_type: str = "uniform",
                      hyperparams: Optional[Dict[str, Dict[str, float]]] = None) -> SigmaPrior:
    """Factory function to create a SigmaPrior (used by pipeline)."""
    if sigma_ranges is None:
        sigma_ranges = get_default_sigma_ranges(pair_types)
    
    if hyperparams is None and prior_type in ['gamma', 'half_cauchy']:
        hyperparams = get_default_hyperparams(prior_type, pair_types)
    
    if gmm_file is not None and os.path.exists(gmm_file):
        return SigmaPrior.from_gmm_file(gmm_file, prior_type, sigma_ranges)
    
    return SigmaPrior(
        pair_types=pair_types,
        sigma_ranges=sigma_ranges,
        prior_type=prior_type,
        use_gmm=False,
        hyperparams=hyperparams
    )