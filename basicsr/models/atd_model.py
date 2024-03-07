import torch

from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class ATDModel(SRModel):

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()
