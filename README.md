**Towards Fast Computation of Certified Robustness
for ReLU Networks**, Tsui-Wei Weng\*, Huan Zhang\*, Hongge Chen, Zhao Song, Cho-Jui Hsieh, Duane Boning, Inderjit S. Dhillon, Luca Daniel (\* Equal Contribution)

[Paper PDF](https://arxiv.org/abs/1804.09699)

In this work, we exploit the special structure of ReLU networks and provide
two computationally efficient algorithms (Fast-Lin and Fast-Lip) that are able
to certify non-trivial lower bounds of minimum distortions, by bounding the
ReLU units with appropriate linear functions (Fast-Lin), or by bounding the
local Lipschitz constant (Fast-Lip).

Prerequisites
-----------------------

The code is tested with python3 and TensorFlow v1.5 and v1.6. We suggest to use Conda to manage your Python environments.
The following Conda packages are required:

```
conda install pillow numpy scipy pandas tensorflow-gpu h5py
conda install --channel numba llvmlite numba
grep 'AMD' /proc/cpuinfo >/dev/null && conda install nomkl
```

You will also need to install Gurobi and its python bindings if you want to try the LP based methods. 

After installing prerequisites, clone our repository:

```
git clone https://github.com/huanzhang12/CertifiedReLURobustness.git
cd CertifiedReLURobustness
```

Our pretrained models can be download here:

```
wget http://jaina.cs.ucdavis.edu/datasets/adv/relu_verification/models_relu_verification.tar
tar xvf models_relu_verification.tar
```

This will create a `models` folder.

How to Run
--------------------

We have provided an interfacing script, `run.sh` to run our code.

```
Usage: ./run.sh model modeltype layer neuron norm solver target
```

* model: mnist or cifar
* modeltype: vanilla (undefended), distilled (Denfensive Distillation), adv\_retrain (Adversarially trained model)
* layer: number of layers (2,3,4 for MNIST and 5,6,7 for CIFAR)
* neuron: number of neurons for each layer (20 or 1024 for MNIST, 2048 for CIFAR)
* norm: p-norm, 1,2 or i (infinity norm)
* solver: ours (Fast-Lin), lip (Fast-Lip), lp (LP)
* target: least, top2 (runner up), random, untargeted

The main interfacing code is `main.py`, which provides additional options. Use `python main.py -h` to explore these options.

Examples
----------------

For example, to evaluate the Linf robustness of MNIST 3\*[1024] adversarially trained model using Fast-Lin on least likely targets, run

```
./run.sh mnist adv_retrain 3 1024 i ours least
```

A log file will be created in the `logs` folder. The last line of the log (starting with [L0]) will report the average
robustness lower bounds on 100 MNIST test images. Lines starting with [L1] reports per-image information.

```
 tail logs/mnist/3/mnist_adv_retrain_3_1024_Li_ours_least_none_*.log
```

```
[L0] model = models/mnist_3layer_relu_1024_adv_retrain, avg robustness_gx = 0.20129, numimage = 96, total_time = 75.3498
```

The adversarially trained model (with adversarial examples crafted by PGD with eps = 0.3) has a robustness lower bound of 0.20129.

Similarly, to evaluate the L1 robustness of MNIST 3\*[20] model on random targets using Fast-Lip, run the following command:

```
./run.sh mnist vanilla 3 20 1 lip random
```

The following result in log file is obtained:

```
[L0] model = models/mnist_3layer_relu_20_best, avg robustness_gx = 2.81436, numimage = 94, total_time = 4.9295
```


Other notes
-------------------

Note that in our experiments we set the number of threads to 1 for a fair comparison to other methods.
To enable multithreaded computing, changing the number `1` in `run.sh` to the number of cores in your system.

```
NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

The code is currently in draft status and there are some unused code and
unclear comments. We are still working on cleaning up the code and improving readability.
You are welcome to create an issue or pull request to report any issues with our code.

