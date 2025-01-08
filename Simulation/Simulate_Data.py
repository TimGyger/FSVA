import torch
import gpytorch

def simulate_gp_response(
    likelihood_type="gaussian", 
    sample_size=1000, 
    nugget=0.1, 
    marginal_variance=1.0, 
    custom_lengthscales=None, 
    seed=1
):
    """
    Simulate response and input variables from a Gaussian Process.

    Parameters:
    - likelihood_type: str, "gaussian" or "bernoulli-logit" for the likelihood.
    - sample_size: int, number of samples to generate.
    - nugget: float, noise variance (nugget).
    - marginal_variance: float, marginal variance (outputscale).
    - custom_lengthscale: list or tensor, custom length scales for each input dimension.
    - seed: int, random seed for reproducibility.

    Returns:
    - X: torch.Tensor, input variables.
    - sampled_field: torch.Tensor, simulated response.
    """

    torch.manual_seed(1)

    # Step 1: Define the Gaussian Process Model
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])  # Enable ARD
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Step 2: Dummy Training Data
    train_x = torch.rand(100, custom_lengthscales.shape[1])  # Small dummy train set
    train_y = torch.zeros(100)    # Dummy target values

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
    model = ExactGPModel(train_x, train_y, likelihood)

    # Step 3: Set Custom Length Scales
    model.covar_module.base_kernel.lengthscale = custom_lengthscales

    # Adjust Marginal Variance (Outputscale)
    model.covar_module.outputscale = marginal_variance

    # Adjust Nugget (Noise Variance)
    if nugget is not None and likelihood_type == "gaussian":
        likelihood.noise = nugget  

    # Step 4: Switch to Evaluation Mode
    model.eval()
    likelihood.eval()

    # Step 5: Generate a Large Input Dataset
    n_points = sample_size  # Large dataset
    input_dim = custom_lengthscales.shape[1]
    X = torch.rand(n_points, input_dim)

    # Step 6: Sample from the GP Prior Without Constructing Full Covariance
    with torch.no_grad():
        latent_values = model(X).sample()

        if likelihood_type == "gaussian":
            sampled_field = latent_values
            probs = None  # No probabilities for Gaussian
        elif likelihood_type == "bernoulli-logit":
            # Convert latent values to probabilities using the sigmoid function
            probs = torch.sigmoid(latent_values)
            # Simulate binary responses based on probabilities
            sampled_field = torch.bernoulli(probs)
        else:
            raise ValueError("likelihood_type must be 'gaussian' or 'bernoulli-logit'.")
        
    return X, sampled_field, probs
