import omegaconf
from math import prod

omegaconf.OmegaConf.register_new_resolver('sum', lambda *x: sum(float(el) for el in x))
omegaconf.OmegaConf.register_new_resolver('prod', lambda *x: prod(float(el) for el in x))
omegaconf.OmegaConf.register_new_resolver('intprod', lambda *x: int(prod(float(el) for el in x)))
