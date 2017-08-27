import torch.nn as nn

class AlexNet(nn.Module):
	def __init__(self, num_classes=2):
		super(AlexNet, self).__init__()

		self.features = nn.Sequential(
				nn.Conv2d(3, 48, kernel_size=11, stride=4),
				nn.ReLU(inplace=True),
				nn.BatchNorm2d(48, eps=0.001),
				nn.MaxPool2d(kernel_size=3, stride=2),
				nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2),
				nn.ReLU(inplace=True),
				nn.BatchNorm2d(128, eps=0.001),
				nn.MaxPool2d(kernel_size=3, stride=2),
				nn.Conv2d(128, 192, kernel_size=3, stride=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(192, 192, kernel_size=3, stride=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(192, 128, kernel_size=3, stride=1),
				nn.ReLU(inplace=True),
				nn.BatchNorm2d(128, eps=0.001),
				nn.MaxPool2d(kernel_size=3, stride=2)
			)

		self.classifier = nn.Sequential(
				nn.Dropout(),
				nn.Linear(128 * 6 * 6, 1024),
				nn.ReLU(inplace=True),
				nn.Dropout(),
				nn.Linear(1024, 1024),
				nn.ReLU(inplace=True),
				nn.Linear(1024, num_classes)
			)

	def forward(self, x):
		x = self.features(x)
		x = x.view(-1, 128 * 6 * 6)
		x = self.classifier(x)
		return x
