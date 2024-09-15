import os 
import requests
import zipfile
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_pizza_steak_sushi():
    data_path = Path("../data")
    image_path = data_path / "pizza_steak_sushi"

    if image_path.is_dir():
        print(f"Path exit")
    else:
        print(f"{image_path} missing ,creating one")
        image_path.mkdir(parents=True,exist_ok=True)

    with open(data_path/ "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    with zipfile.ZipFile(data_path/ "pizza_steak_sushi.zip",'r') as zip_file:
        print(f"unzip file")
        zip_file.extractall(image_path)
    os.remove(data_path/ "pizza_steak_sushi.zip")


def create_dataloaders(train_dir:str, 
                       test_dir:str,
                       transform:transforms.Compose,
                        batch_size:int):
    numworkers = os.cpu_count()
    print(f"loading data use {numworkers} cpus")
    train_data = datasets.ImageFolder(train_dir,transform=transform)
    test_data = datasets.ImageFolder(test_dir,transform=transform)
    class_name = train_data.classes
    train_dataloader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=numworkers)
    test_dataloader = DataLoader(dataset=test_data,batch_size=batch_size,num_workers=numworkers)
    return train_dataloader,test_dataloader,class_name


if __name__ == '__main__':
    prepare_pizza_steak_sushi()