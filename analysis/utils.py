import polars as pl
import numpy as np


def large_print(ds):
    with pl.Config(tbl_rows=ds.height, tbl_cols=ds.width):
        print(ds)


#  def bootstrap_std_of_mean(nums, n_bootstrap_samples=1000):
#      # Array to store the mean of each bootstrap sample
#      bootstrap_means = np.zeros(n_bootstrap_samples)
#
#      # Generate bootstrap samples and compute their means
#      for i in range(n_bootstrap_samples):
#          bootstrap_sample = np.random.choice(nums, size=len(nums), replace=True)
#          bootstrap_means[i] = np.mean(bootstrap_sample)
#
#      # The standard deviation of the bootstrap means is an estimate of the std of the original mean
#      std_of_mean = np.std(bootstrap_means)
#      return std_of_mean


def mu_sigma_std(nums):
    nums = np.array(nums)
    n = len(nums)
    mu = nums.mean()
    sigma = nums.std()
    std = sigma / np.sqrt(n)
    #  std = bootstrap_std_of_mean(nums, n_bootstrap_samples=100)
    return mu, sigma, std


def mu_sigma_std_from_sums(sums, nums):
    #  $X\sim N(\mu,\sigma)$
    #  $S_i=\sum_{j=1}^{n_i} X_{i,j}$
    #  best mean:
    #  $\hat{\mu}=\frac{\sum_{i=1}^m S_m}{\sum_{i=1}^m n_i}
    #      variance:
    #          $Var(\hat{\mu})=\frac{\sigma^2}{\sum_{i=1}^m n_i}$
    #  best variance:
    #      $\hat{\sigma^2}=\frac{1}{m-1}[\sum_{i=1}^m\frac{S_i^2}{n_i}-\frac{(\sum_{i=1}^m S_i)^2}{\sum_{i=1}^m n_i}]$
    #      $\hat{\sigma^2}=\frac{1}{m-1}[\sum_{i=1}^m n_i(\frac{S_i}{n_i})^2-\sum_{i=1}^m n_i(\frac{\sum_{i=1}^m S_i}{\sum_{i=1}^m n_i})^2]$
    #      $\hat{\sigma^2}=\frac{1}{m-1}[\sum_{i=1}^m n_i(\frac{S_i}{n_i})^2-\sum_{i=1}^m n_i\hat{\mu}^2]$
    assert len(sums) == len(nums)
    sums = np.array(sums)
    nums = np.array(nums)
    mask = nums > 0
    sums = sums[mask]
    nums = nums[mask]
    #  try:
    #      import warnings
    #
    #      warnings.filterwarnings("error")
    #
    mu = sums.sum() / nums.sum()
    #  except RuntimeWarning:
    #      print("!!!!!!!!", sums, nums)
    #      raise
    #  if inf shows up in sums, all inf
    if not np.all(np.isfinite(sums)):
        return np.inf, np.inf, np.inf
    sigma2 = ((sums**2 / nums).sum() - nums.sum() * mu**2) / (len(sums) - 1)
    sigma = np.sqrt(sigma2)
    std = np.sqrt(sigma2 / nums.sum())
    return mu, sigma, std


#
#  def ration_mu_std(mu1, std1, mu2, std2):
#      mu = mu1 / mu2
#      std = mu * np.sqrt((std1 / mu1) ** 2 + (std2 / mu2) ** 2)
#      return mu, std
#
#
#  def minus_mu_std(mu1, std1, mu2, std2):
#      mu = mu1 - mu2
#      #  std = np.sqrt(std1**2 + std2**2)
#      std = std1 + std2
#      return mu, std


def format_mu_std(mu, std, digit=None, latex=False, zero_std_digit=1):
    if std == 0.0:
        digit = zero_std_digit
    if mu == -0.0:
        mu = 0.0
    if digit is None:
        # infer_digit based on std
        # if std starts with 1, keep another digit
        # if std starts with 2 or more, keep to that digit
        first_digit = -int(np.log10(std)) + 1
        if std * 10**first_digit >= 2:
            effective_digit = first_digit
        else:
            effective_digit = first_digit + 1
        digit = max(0, effective_digit)
    if not latex:
        s = f"{mu:.{digit}f}±{std:.{digit}f}"
    else:
        s = f"${mu:.{digit}f}\pm{std:.{digit}f}$"
    return s
