from __future__ import print_function, division
import os
import sys
import argparse
import enum
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

name = {'0000':'one','0001':'two','0002':'three','0003':'four','0004':'five','0005':'six','0006':'seven','0007':'eight','0008':'nine'}
namerafdb = {'1':'C1','2':'C2','3':'C3','4':'C4','5':'C5','6':'C6','7':'C7'}
DatasetsLocation = {
	'raf-db':{7: 'Train'}
}


DatasetsMeanAndStd = {
	'raf-db':{
		7: {'mean': [0.5754, 0.4499, 0.4017], 'std':  [0.2636, 0.2403, 0.2388]}
	}
	

}


class ClassesRAFDB(enum.Enum):
    C1 = 0
    C2 = 1
    C3 = 2
    C4 = 3
    C5 = 4
    C6 = 5
    C7 = 6
    



classes_map = {
	"raf-db": {7: ClassesRAFDB}
	
}


def default_loader(path, transform):
	return transform(Image.open(path))

def video_frames_loader(path, transform):
	images = []
	for image in os.listdir(path):
		images.append(transform(Image.open(os.path.join(path,image))))
	return torch.stack(images,dim=1)
		
def image_loader(path, transform):
    return transform(Image.open(path))


class ClassificationDataset(Dataset):

	def __init__(self, args, split, train, arch_split_file):
		super().__init__()
		self.base_dir = DatasetsLocation.get(args.dataset).get(args.no_of_classes)
		if split is not None:
			self.base_dir = os.path.join(self.base_dir, split)
		else:
			self.image_dir = os.path.join(args.data, args.dataset.upper(), self.base_dir)
		self.image_paths = []
		self.labels = []
		self.loader = image_loader
		val_subject = args.subject
		
		self.transform = transforms.Compose([])
		if args.dataset == 'raf-db' 
			self.transform.transforms.append(transforms.Resize((120, 120)))
		
		self.transform.transforms.append(transforms.ToTensor())
		self.transform.transforms.append(
			transforms.Normalize(DatasetsMeanAndStd.get(args.dataset).get(args.no_of_classes).get('mean'),
								  DatasetsMeanAndStd.get(args.dataset).get(args.no_of_classes).get('std')))


		print(f'Number Of Videos in {args.dataset.upper()} dataset split: {arch_split_file}: {len(self.image_paths)}')

	def __getitem__(self, index):
		path = self.image_paths[index]
		label = self.labels[index]
		img = self.loader(path, self.transform)
		return {'image': img, 'label': label}

	def __len__(self):
		return len(self.image_paths)


if __name__ == '__main__':
	parser = argparse.ArgumentParser("c")
	args = parser.parse_args()
	args.data ="data"
	args.dataset = "raf-db"
	args.no_of_classes = 7
	args.subject = None
	train_data = ClassificationDataset(args, None, train=True,arch_split_file='dataset_files/'+args.dataset+'/train.txt')
	print(train_data.__getitem__(0)['image'].size())
