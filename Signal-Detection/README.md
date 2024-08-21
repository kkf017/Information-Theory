### Signal detection

Set the following problem:

    Ho: x[n] = w[n] with n = 0,....,N-1 (noise)

    H1: x[n] = s[n] + w[n] with n = 0,....,N-1 (signal + noise)

with s = 11,...,1  s[0], s[1], ...,s[N-1] = 1, and w ~ N(mu, sigma) with $\mu$ = 0 and $\sigma$ = 1.

Verify H1 if:

$$\left( \sum_{n=0}^{N-1} x[n]s[n] \right) \gt  ln(\gamma) \sigma² + \frac{1}{2} \left( \sum_{n=0}^{N-1} s[n]²\right)$$

Verify H0 if:
    
$$\left( \sum_{n=0}^{N-1} x[n]s[n] \right) \lt  ln(\gamma) \sigma² + \frac{1}{2} \left( \sum_{n=0}^{N-1} s[n]²\right)$$
