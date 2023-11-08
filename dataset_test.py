
from __future__ import print_function, division
import os
import argparse
import enum
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

name = {'0000':'one','0001':'two','0002':'three','0003':'four','0004':'five','0005':'six','0006':'seven','0007':'eight','0008':'nine'}
namerafdb = {'1':'C1','2':'C2','3':'C3','4':'C4','5':'C5','6':'C6','7':'C7'}
DatasetsLocation = {
	'raf-db':{7: 'Test'}
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

	def __init__(self, args, split, train):
		super().__init__()
		self.base_dir = DatasetsLocation.get(args.dataset).get(args.no_of_classes)
		if split is not None:
			self.base_dir = os.path.join(self.base_dir, split)
		if "oulu" in args.dataset:
			self.image_dir = os.path.join(args.data, "OULU", self.base_dir)
		else:
			self.image_dir = os.path.join(args.data, args.dataset.upper(), self.base_dir)
		self.image_paths = []
		self.labels = []
		self.loader = image_loader
		val_subject = args.subject
		# subjects_or_folds = os.listdir(self.image_dir)
		classes = os.listdir(self.image_dir)
		# if val_subject is not None:
		# 	if train:
		# 		del subjects_or_folds[val_subject]
		# 	else:
		# 		subjects_or_folds = [subjects_or_folds[val_subject]]
		# if split is not None:
		# 	label = 0
		# 	for class_folder in os.listdir(os.path.join(self.image_dir)):
		# 		for image in os.listdir(os.path.join(self.image_dir, class_folder)):
		# 			address = os.path.join(self.image_dir, class_folder, image)
		# 			self.image_paths.append(address)
		# 			self.labels.append(label)
		# 		label += 1
		# else:
		class_videos_address = [[] for _ in range(args.no_of_classes)]
		for i in range(0, len(classes)):
			class_folder = classes[i]
			label = 0
			# for class_folder in os.listdir(os.path.join(self.image_dir, subject)):
			for image in os.listdir(os.path.join(self.image_dir, class_folder)):
				address = os.path.join(self.image_dir, class_folder, image)
				print(address)
				self.image_paths.append(address)
				if args.dataset == "disfa" or args.dataset == "mmi":
					self.labels.append(classes_map.get(args.dataset).get(args.no_of_classes)[class_folder.split("_")[1]].value)
				
				elif args.dataset == 'raf-db':
					print(classes_map.get(args.dataset).get(args.no_of_classes)[namerafdb[class_folder]].value)
					self.labels.append(classes_map.get(args.dataset).get(args.no_of_classes)[namerafdb[class_folder]].value)
					class_videos_address[classes_map.get(args.dataset).get(args.no_of_classes)[namerafdb[class_folder]].value].append(address)
						
				
				else:
					self.labels.append(label)
			label += 1
			
			# print("subject",subject,"no_of_videos",no_of_videos)

#         if abs(((len(self.image_paths) // args.batch_size) * args.batch_size) - len(self.image_paths)) == 1 and len(self.image_paths) != 1:
#             del self.image_paths[-1]
#             del self.labels[-1]
		path1 = os.path.join("dataset_files", args.dataset,"test.txt")
		# path2 = os.path.join("dataset_files", args.dataset,"validation.txt")

		for i in range(0,len(class_videos_address)):
			a = len(class_videos_address[i])
			# x = a - a//5
			with open(path1, 'a') as f:
			  for item in class_videos_address[i][0:a]:
				   f.write(f"path: {item} label:{i}\n")
			# with open(path2, 'a') as f:
			#    for item in class_videos_address[i][x//2:x]:
			# 	   f.write(f"path: {item} label:{i}\n")
			# with open(path3, 'a') as f:
			#    for item in class_videos_address[i][x:]:
			# 	   f.write(f"path: {item} label:{i}\n")
		print(len(class_videos_address[0]),len(class_videos_address[1]),len(class_videos_address[2]),len(class_videos_address[0]))

		self.transform = transforms.Compose([])
		if args.dataset == 'raf-db':
			self.transform.transforms.append(transforms.Resize((120, 120)))
		

		self.transform.transforms.append(transforms.ToTensor())
		images = [] 
		for path in self.image_paths:	
		   images.append(self.loader(path, self.transform))
			
		channel_1 =[]
		channel_2 =[]
		channel_3 =[]
		for i in range(0,len(images)):
			channel_1.append(images[i][0])
			channel_2.append(images[i][1])
			channel_3.append(images[i][2])
		
		print("channel_1",torch.mean(torch.stack(channel_1)), torch.std(torch.stack(channel_1)))
		print("channel_2",torch.mean(torch.stack(channel_2)), torch.std(torch.stack(channel_2)))
		print("channel_3",torch.mean(torch.stack(channel_3)), torch.std(torch.stack(channel_3)))
		# print(f'Number Of Videos in {args.dataset.upper()} and Subjects: {len(subjects_or_folds)} dataset : {len(self.image_paths)}')

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
	train_data = ClassificationDataset(args, None, train=True)
	print(train_data.__getitem__(0)['image'].size())
