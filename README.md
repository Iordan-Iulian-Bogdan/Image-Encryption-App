This is an implementation of an encryption algorithm for images using compressive sensing, this is meant to be more of a proof of concept where we use compressive sensing in a novel way to encrypt images.

The overall algorthm for this is as follows : 

```
Inputs : 𝜓 measurement matrix, 𝑥 vectorized image, 𝑁 signal length of 𝑥, 𝑘number of iterations for ADM, 𝜏, 𝛽 parameters for ADM, 𝑒𝑘, θ auxiliary vectors

𝑦 ← 𝜓𝑥 // extracting the measurements, this effectively encrypts the signal x into y

For 𝑖 ← 0 până la N , 𝑖 ← 𝑖 + 1 execute // construct dictionary 𝐴
  𝑒𝑘 = 0
  𝑒𝑘 𝑖 = 1
  θ = 𝐼𝐷𝐶𝑇(𝑒𝑘)
  𝐴 : , 𝑖 = 𝜓 ∗ θ
End For

𝑠 = 𝐴𝐷𝑀(𝐴, 𝑦, 𝜏,𝛽, 𝑘) // solves the A * x = y equation, where x is the unknown encrypted signal

For  𝑖 ← 0 până la N , 𝑖 ← 𝑖 + 1 execute // reconstruct signal x which now represents the decrypted signal
  𝑒𝑘 = 0
  𝑒𝑘 𝑖 = 1
  θ = 𝐼𝐷𝐶𝑇(𝑒𝑘)
  𝑥′ = 𝑥′ + θ ∗ 𝑠(𝑖)
Sf - End

Outputs : 𝑥′ vectorized decrypted image
```
