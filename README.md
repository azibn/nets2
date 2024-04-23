# nets2

`nets2` is a project using Convolutional Neural Networks (CNN) to characterise exocomet transits in photometric data. The method is based on Feinsetin et al. 2020, who developed this framework and applied it to the context of stellar flares. We adapted it to the needs of our project. The differences between this repository and [stella](https://github.com/afeinstein20/stella) lie in the setup of the architecture. The running of the CNN is operated in the same way.

To install with mamba:

`mamba create -f env.yml`

or appropriately replace `mamba` with your chosen package manager. You can also do:

```
git clone https://github.com/azibn/nets2
cd stella
python setup.py install
```

`env.yml` was tested with an Apple Silicon machine.

