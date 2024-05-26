This is an implementation of an encryption algorithm for images using compressive sensing, this is meant to be more of a proof of concept where we use compressive sensing in a novel way to encrypt images.

The overall algorthm for this is as follows : 

```
Inputs : 𝜓 measurement matrix, 𝑥 vectorized image to be encrypted, 𝑁 signal length of 𝑥, 𝑘number of iterations for ADM, 𝜏, 𝛽 parameters for ADM, 𝑒𝑘, θ auxiliary vectors

// encryption step
𝑦 ← 𝜓𝑥 // extracting the measurements, this effectively encrypts the signal 𝑥 into y

//decryption step
For 𝑖 ← 0 to N , 𝑖 ← 𝑖 + 1 execute // construct dictionary 𝐴
  𝑒𝑘 = 0
  𝑒𝑘 𝑖 = 1
  θ = 𝐼𝐷𝐶𝑇(𝑒𝑘)
  𝐴 : , 𝑖 = 𝜓 ∗ θ
End For

𝑠 = 𝐴𝐷𝑀(𝐴, 𝑦, 𝜏,𝛽, 𝑘) // solves the 𝐴 * 𝑥 = 𝑦 equation, where 𝑥 is the unknown encrypted signal using the Alternating Direction Method algorithm

For  𝑖 ← 0 to N , 𝑖 ← 𝑖 + 1 execute // reconstruct signal 𝑥 which now represents the decrypted signal
  𝑒𝑘 = 0
  𝑒𝑘 𝑖 = 1
  θ = 𝐼𝐷𝐶𝑇(𝑒𝑘)
  𝑥′ = 𝑥′ + θ ∗ 𝑠(𝑖)
Sf - End

Outputs : 𝑥′ vectorized decrypted image
```

This works only if the  measurement matrix ```𝜓``` is identical upon encryption and decryption. ```𝜓``` is usually meant to be a random matrix but by using a deterministic number generator which is seeded using a passphrase we can encrypt and decrypt an arbitrary signal. Keep in mind that this method is not lossless, the reconstructed signal will not be 100% identical, this is why you'd only want to use something like this for things like images.

Because ```𝑥``` is a vectorized image which means it can have millions of elements the dictionary ```A``` is going to be a matrix with potentially billions of elements (so dozens of GB in size). The challenge in doing something like this comes from the fact that the matrices involved occupy so much memory that it's impossible to solve this problem on a regular computer as is, however, we can divide the original image in smaller chunks that can fit in the memory of a typical computer.

Another issue is speed, compressive sensing is extremely costly computationally as well, which is why this encryption method is GPU accelerated using OpenCL.

How to use :

```encryptionImage img_encrypted = encryptImage(img, TILE_SIZE, "5v48v5832v5924", 4);```

This encrypts the image and stores it into a struct which contains the encrypted data and some other info needed to properly decrypt the image after. 

```
encryptionImage encryptImage(cv::Mat img, /* image to be encrypted */
							int TILE_SIZE, /* size of tiles in which the image is broken up and processed, larger tiles may provide better quality at the cost of memory and speed */
							string passphrase, /* passphare used to generate the encryption matrix */
							int threads /* number of tiles to be encrypted simultaneously */ )
```

The longer the passphrase the more seeds are used to generate the encryption matrix ```A``` and thus the encryption is more secure. 

```TILE_SIZE``` this is the size of the chunks that are going to processed at a time, larger tile sizes will use more memory but might produce better results. A GPU with more than 8GB of VRAM is needed for a tile size of 128 for example.

```threads``` is the ammount of instances of chunks of the image that are encrypted in parallel, might provide a speed up, sepcially if the TILE_SIZE is smaller.

Performance : 

~25 minutes to decrypt a 1024x576 image on a 7900 XT.

TODO:
- [ ] Add CPU fallback
- [ ] hybrid CPU & GPU acceleration 
- [ ] FP16 for potentially faster execution on GPU
- [ ] General optimizations
