from transformers import AutoModel
from ULTRA.ultra.datasets import CoDExSmall
from ULTRA.ultra.eval import test
model = AutoModel.from_pretrained("mgalkin/ultra_4g", trust_remote_code=True)
dataset = CoDExSmall(root="./datasets/")
test(model, mode="test", dataset=dataset, gpus=None)