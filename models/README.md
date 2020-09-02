The `pcb_model.zip`, `fpb_model.zip` and `heteroencoder_model.zip` must be loaded as `ddc_pub.ddc_v3.DDC` models.
```python
from ddc_pub import ddc_v3 as ddc
pcb_model = ddc.DDC(model_name="models/pcb_model") #without extension
fpb_model = ddc.DDC(model_name="models/fpb_model")
het_model = ddc.DDC(model_name="models/heteroencoder_model")
```

The `prior_model.zip` and `tl_model.zip` must be loaded as `ddc_pub.ddc_v3_unbiased.DDC` models. *Unbiased* stands for unbiased decoder, i.e. a decoder for which we don't initialize the states according to descriptors.
```python
from ddc_pub import ddc_v3_unbiased as ddc_unbiased
prior_model = ddc_unbiased.DDC(model_name="models/prior_model")
tl_model = ddc_unbiased.DDC(model_name="models/tl_model")
```

The `qsar_model.pickle` should be loaded as:
```python
import pickle
with open("models/qsar_model.pickle", 'rb') as file:
        model_dict = pickle.load(file)
drd2_model = model_dict["classifier_sv"]
```
