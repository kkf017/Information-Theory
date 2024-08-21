## Signal detection

Set the following problem:

    Ho: x[n] = w[n] with n = 0,....,N-1 (noise)

    H1: x[n] = s[n] + w[n] with n = 0,....,N-1 (signal + noise)

with s = 11,...,1  s[0], s[1], ...,s[N-1] = 1, and w ~ N($\mu$, $\sigma$²) with $\mu$ = 0 and $\sigma$ = 1.

Verify H1 if:

$$\left( \sum_{n=0}^{N-1} x[n]s[n] \right) \gt  ln(\gamma) \sigma² + \frac{1}{2} \left( \sum_{n=0}^{N-1} s[n]²\right)$$

Verify H0 if:
    
$$\left( \sum_{n=0}^{N-1} x[n]s[n] \right) \lt  ln(\gamma) \sigma² + \frac{1}{2} \left( \sum_{n=0}^{N-1} s[n]²\right)$$


### Demonstration

**Neyman-Pearson rule** :
$$\left( \frac{P(x;H1)}{P(x;H0)} \right) \gt  \gamma $$

**Baysian rule** :
$$\left( \frac{P(x|H1)}{P(x|H0)} \right) \gt  \frac{P(H0)}{P(H1)} $$ (with very often P(H0) = P(H1))

$$ P(x;H1) = \frac{1}{(2\pi\sigma²)^{N/2}} \exp{(-\frac{1}{2\sigma²} \sum_{n=0}^{N-1} (x[n]-s[n])²)}$$

$$ P(x;H0) = \frac{1}{(2\pi\sigma²)^{N/2}} \exp{(-\frac{1}{2\sigma²} \sum_{n=0}^{N-1} x[n]²)}$$


Apply Neyman-Pearson rule :

$$ \left( \frac{P(x;H1)}{P(x;H0)} \right) \gt  \gamma $$

$$  exp{(-\frac{1}{2\sigma²} ((x-s)^T(x-s) - x^Tx) )}  \gt  \gamma $$

$$  -\frac{1}{2\sigma²} ((x-s)^T(x-s) - x^Tx)   \gt  ln(\gamma) $$

$$  - (\sum_{n=0}^{N-1} s[n]² - 2\sum_{n=0}^{N-1} x[n]s[n])   \gt  ln(\gamma)2\sigma² $$

$$  - s^Ts + 2x^Ts   \gt  ln(\gamma)2\sigma² $$

$$ x^Ts   \gt  ln(\gamma)2\sigma² + \frac{1}{2} s^Ts$$

$$ \left( \sum_{n=0}^{N-1} x[n]s[n] \right) \gt  ln(\gamma) \sigma² + \frac{1}{2} \left( \sum_{n=0}^{N-1} s[n]²\right) $$
