from torchvision import transforms
from data.augmentations.cut_out import *
from data.augmentations.randaugment import RandAugment


def get_transform(transform_type='default', image_size=32, args=None):
    if transform_type == 'domainnet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        train_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])


    elif transform_type == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        train_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    elif transform_type == 'orca':
        from project_utils.cluster_utils import TransformTwice
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transform = TransformTwice(transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
                ]))
        test_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
                ])


    elif transform_type == 'cutout':

        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])

        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            normalize(mean, std),
            cutout(mask_size=int(image_size / 2),
                   p=1,
                   cutout_inside=False),
            to_tensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif transform_type == 'rand-augment':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_transform.transforms.insert(0, RandAugment(args.rand_aug_n, args.rand_aug_m, args=None))

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    else:
        raise NotImplementedError

    return (train_transform, test_transform)