import torch
from torchvision import datasets
from torchvision import transforms
from torch import nn
from torch import optim
from torchvision.utils import save_image
import os
import csv

from generator import Generator
from discriminator import Discriminator

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/',
                          download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)

# create instances of the generator and discriminator
z_dim = 100
gen = Generator(z_dim)
disc = Discriminator()

# set device to GPU or CPU based on availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = gen.to(device)
disc = disc.to(device)

# define the loss function
criterion = nn.BCELoss()

# define the optimizers
lr = 0.0002
beta1 = 0.5
gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))
disc_opt = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))

# smoothing parameter for label smoothing
smooth = 0.1


def train(epochs=50, z_dim=100):
    # Load the last checkpoint if it exists
    start_epoch = 0
    if os.path.isfile('gan_checkpoint.pt'):
        checkpoint = torch.load('gan_checkpoint.pt')
        start_epoch = checkpoint['epoch']
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])
        disc_opt.load_state_dict(checkpoint['disc_opt_state_dict'])

    total_epochs = start_epoch + epochs

    # open log files
    with open("C:/coding/dataStructures/GAN-NI/data/training_log.txt", "a") as log_file,\
            open("C:/coding/dataStructures/GAN-NI/data/training_log.csv", "a", newline='') as csv_file:

        fieldnames = ['epoch', 'gen_loss', 'disc_loss']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Define training loop
        for epoch in range(start_epoch, total_epochs):
            for real, _ in trainloader:
                cur_batch_size = len(real)
                real = real.view(cur_batch_size, -1).to(device)

                # Train Discriminator
                disc_opt.zero_grad()

                # Generate fake images
                noise = torch.randn(cur_batch_size, z_dim).to(device)
                fake = gen(noise)

                disc_real = disc(real).view(-1)
                disc_fake = disc(fake.detach()).view(-1)

                # Apply label smoothing
                real_labels = torch.full(
                    (cur_batch_size,), 1 - smooth, device=device)
                fake_labels = torch.full(
                    (cur_batch_size,), smooth, device=device)

                disc_loss = criterion(disc_real, real_labels) + \
                    criterion(disc_fake, fake_labels)

                # calculate accuracy
                real_correct = (disc_real > 0.5).sum().item()
                fake_correct = (disc_fake < 0.5).sum().item()
                disc_accuracy = (real_correct + fake_correct) / \
                    (2 * cur_batch_size)

                # Backpropagation
                disc_loss.backward()
                disc_opt.step()

                # Train Generator
                gen_opt.zero_grad()
                gen_fake = disc(fake)
                gen_loss = criterion(gen_fake, torch.ones_like(gen_fake))

                # Backpropagation
                gen_loss.backward()
                gen_opt.step()

            # Save model every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch + 1 == total_epochs:
                torch.save({
                    'epoch': epoch + 1,  # save as next epoch since we've finished the current one
                    'gen_state_dict': gen.state_dict(),
                    'disc_state_dict': disc.state_dict(),
                    'gen_opt_state_dict': gen_opt.state_dict(),
                    'disc_opt_state_dict': disc_opt.state_dict(),
                }, "gan_checkpoint.pt")

                # Save some generated images for inspection
                save_image(fake.view(fake.size(0), 1, 28, 28),
                           f"C:/coding/dataStructures/GAN-NI/output/generated_images_epoch_{epoch + 1}.png")

                log_line = (
                    f"Epoch {epoch + 1}/{total_epochs}:\t"
                    f"Generator loss: {gen_loss},\t"
                    f"Discriminator loss: {disc_loss}\n"
                )

                print(log_line, end="")
                log_file.write(log_line)

                # Log the losses to CSV
                writer.writerow(
                    {'epoch': epoch + 1, 'gen_loss': gen_loss.item(), 'disc_loss': disc_loss.item()})


# call the training function
train(epochs=50, z_dim=100)
