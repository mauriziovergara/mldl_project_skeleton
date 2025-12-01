# Function to denormalize image for visualization
def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

# Visualize 10 random examples from the dataset
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten() # Flatten the 2x5 array of axes for easier iteration
classes_sampled = []
found_classes = 0

for inputs, labels in dataloader_train:
  for img, label in zip(inputs, labels):
    class_name = tiny_imagenet_dataset_train.classes[label]
    if class_name not in classes_sampled:
      ax = axes[found_classes]
      ax.imshow(denormalize(img))
      ax.set_title(class_name)
      ax.axis('off')
      classes_sampled.append(class_name)
      found_classes += 1
    if found_classes == 10:
      break
  if found_classes == 10:
    break

plt.tight_layout()
plt.show()
