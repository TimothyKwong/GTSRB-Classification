from torch.utils.data import Dataset
from PIL import Image
import os
import smote

class RoadSignDataset(Dataset):
    def __init__(self, path_train, path_val, transform, smote_state=False):
        self.path_train = path_train
        self.path_val = path_val
        self.trainset = []
        self.valset = []
        self.transform = transform
        self.smote_state = smote_state
        self.trainSample_dist = []
        self.smoteSample_dist = []

    def get_images(self):
        # get training + validation path directory
        path_train = [f.path for f in os.scandir(self.path_train) if f.is_dir()]
        path_val = [f.path for f in os.scandir(self.path_val) if f.is_dir()]
        num_of_classes = len(path_train)

        # make training set
        for num_of_subfolders in range(num_of_classes):
            a0 = 0
            for image_path in os.listdir(path_train[num_of_subfolders]):
                concat_path = os.path.join(path_train[num_of_subfolders], image_path)
                image = Image.open(concat_path)
                label = num_of_subfolders

                t0_image = self.transform[0]( image )
                self.trainset.append((t0_image, label))
                a0+=1

            self.trainSample_dist.append([a0])

        # make smote set
        if self.smote_state == True:
            s = smote.SMOTE(0, self.path_train, self.path_val)
            for i in range(num_of_classes):
                s1 = 0
                for image_path in s.get_samples()[i]:
                    image = Image.open( image_path[0] )

                    t1_image = self.transform[1]( image )
                    self.trainset.append( (t1_image, image_path[1]) )
                    s1+=1

                self.smoteSample_dist.append([s1])    

        # make validation set
        for num_of_subfolders in range(len(path_val)):
            for image_path in os.listdir(path_val[num_of_subfolders]):
                concat_path = os.path.join(path_val[num_of_subfolders], image_path)
                image = Image.open(concat_path)
                label = num_of_subfolders

                t0_image = self.transform[0](image)
                self.valset.append((t0_image, label))

        return self.trainset, self.valset, num_of_classes, self.trainSample_dist, self.smoteSample_dist

    def __len__(self):
        return len(self.trainset), len(self.valset)
