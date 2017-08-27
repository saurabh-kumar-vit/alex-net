import torch
from torchvision import datasets, transfroms

def load_sample_set():
	transfrom = transfroms.Compose([
			transfroms.Scale(256),
			transfroms.CenterCrop(227),
			transfroms.ToTensor(),
			transfroms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
		])

	sample_set = datasets.ImageFolder('./data/sample', transfrom)
	sample_set_loader = torch.utils.data.DataLoader(	sample_set,
														batch_size=20,
														shuffle=False,
														num_workers=1)

	return sample_set_loader
