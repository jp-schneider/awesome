
import copy
from os import PathLike
from typing import Any, BinaryIO, Dict, Optional, Type, Union
import torch
from awesome.util.reflection import class_name, dynamic_import
from awesome.serialization import JsonConvertible
import copy
from awesome.util.torch import TensorUtil
class PriorCache():

    __cache__: Dict[int, Any]

    def __init__(self, 
                 model_type: Type[torch.nn.Module],
                 model_args: Dict[str, Any],
                 store_device: Optional[torch.device] = None,
                 ) -> None:
        self.__cache__ = {}
        self.model_type = model_type
        self.model_args = copy.deepcopy(model_args)
        if store_device is None:
            store_device = torch.device('cpu')
        self.store_device = store_device

    def __contains__(self, key: int) -> bool:
        return key in self.__cache__
    
    def generate_prior(self, key: int) -> Any:
        # Create a model and take its state dict as a prior
        model = self.model_type(**self.model_args)
        return model.state_dict()
    
    @staticmethod
    def extract_prior(model: torch.nn.Module) -> Any:
        if hasattr(model, 'extract_prior') and callable(model.extract_prior):
            return model.extract_prior()
        sharing_dict = model.state_dict()
        detached_dict = copy.deepcopy(sharing_dict)
        return detached_dict

    @staticmethod
    def apply_prior(model: torch.nn.Module, prior: Any) -> None:
        if hasattr(model, 'apply_prior') and callable(model.apply_prior):
            return model.apply_prior(prior)
        model.load_state_dict(prior)


    def __getitem__(self, key: int) -> Any:
        if key not in self.__cache__:
            self.__cache__[key] = self.generate_prior(key)
        return self.__cache__[key]
        

    def __setitem__(self, key: int, value: Any) -> None:
        if self.store_device is not None:
            value = TensorUtil.apply_deep(value, fnc=lambda x: x.to(device=self.store_device))
        self.__cache__[key] = value


    def get_state(self) -> Dict[str, Any]:
        res = dict()
        res['model_type'] = class_name(self.model_type)
        res['model_args'] = JsonConvertible.convert_to_json_str(self.model_args)
        res['store_device'] = str(self.store_device)
        res['cache'] = {str(k): v for k, v in self.__cache__.items()}
        return res

    def save(self, f: Union[PathLike[str], BinaryIO]) -> None:
        res = self.get_state()
        torch.save(res, f)

    def set_state(self, state: Dict[str, Any]) -> None:
        res = state
        self.model_type = dynamic_import(res['model_type'])
        self.model_args = JsonConvertible.load_from_string(res['model_args'])
        device = res.get('store_device', torch.device('cpu'))
        if device is None or device == 'None':
            device = "cpu"
        self.store_device = torch.device(device)
        self.__cache__ = {int(k): v for k, v in res['cache'].items()}

    @classmethod
    def load(cls, f: Union[PathLike[str], BinaryIO]) -> 'PriorCache':
        args = dict()
        if not torch.cuda.is_available():
            args['map_location']=torch.device('cpu')
        res = torch.load(f, **args)
        prior_cache = PriorCache(None, None)
        prior_cache.set_state(state=res)
        return prior_cache