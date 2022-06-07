from typing import Any

from project.data.common.kfold_data_module import KFoldDataModule
from project.models.common.nlp_configs import HFTransformerDataConfig


class TokenizerDataModule(KFoldDataModule):
    def __init__(self, tokenizer: Any, cfg: HFTransformerDataConfig = HFTransformerDataConfig()) -> None:
        super().__init__(cfg=cfg)
        self.tokenizer = tokenizer
