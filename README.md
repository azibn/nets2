# nets2

Built on `stella` (Feinstein et al. 2020), `nets2` is a Python tool to create and train a neural network repurposed to identify exocomets. This project develops a synthetic exocomet training set, trains a neural network, and feeds in lightcurves to the neural network model. The results are given by returning a probability at each data point corresponding to how likely it is an exocomet.

Since this package is repurposed, the best way to install it is:

```
git clone https://github.com/azibn/nets2.git
cd nets2
mamba create -f env.yml
mamba activate nets2
```

The env.yml file above was tested on an M1 Apple Silicon machine. If you would like make use of the GPU-enabled `tensorflow-metal`, use the `env-metal.yml` file instead.
