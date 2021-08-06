import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset


def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    

    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        val_dataset = CoCoDataset(
            image_dir='/data2/liushilong/data/coco14/val2014', 
            anno_path='/data2/liushilong/data/coco14/annotations/instances_val2014.json',
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)
        
    print("len(val_dataset):", len(val_dataset))
    return val_dataset
