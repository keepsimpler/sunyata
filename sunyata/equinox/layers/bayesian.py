import jax.numpy as jnp
import equinox as eqx

class BayesianLayer(eqx.Module):
    positive_method: str = 'none'
    
    def __init__(self, positive_method:str = 'none'):
        super().__init__()
        self.positive_method = positive_method

    def __call__(self, priors, evidence):
        if self.positive_method == 'abs':
            evidence = jnp.abs(evidence)
        elif self.positive_method == 'square':
            evidence = evidence ** 2
        elif self.positive_method == 'exp':
            evidence = jnp.exp(evidence)
            
        weighted_sum_of_evidence = jnp.sum(priors * evidence, axis=-1, keepdims=True)
        posteriors = (priors * evidence) / weighted_sum_of_evidence
        
        return posteriors


if __name__ == '__main__':
    bayesian_layer = BayesianLayer(positive_method='abs')