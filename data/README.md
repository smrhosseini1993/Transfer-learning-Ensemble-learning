# Data Format

## Required Structure

```
data/
├── training/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   ├── ...
│   └── ica_lables.txt
└── test/
    ├── image_001.jpg
    ├── image_002.jpg
    ├── ...
    └── ica_lables.txt
```

## Image Files

- **Format:** JPG
- **Naming:** Any consistent naming (e.g., `image_001.jpg`, `image_002.jpg`)
- **Important:** Images must be sorted alphabetically to match labels

## Label Files

- **Filename:** `ica_lables.txt`
- **Format:** Plain text, one label per line
- **Values:** `0` (negative class) or `1` (positive class)
- **Order:** Must match alphabetically sorted image filenames

**Example:**
```
0
1
1
0
```

## Checklist

- [ ] Images are JPG format
- [ ] Images in `data/training/` and `data/test/`
- [ ] Label files named `ica_lables.txt`
- [ ] Number of labels = number of images
- [ ] Labels are 0 or 1 only
- [ ] No extra lines or spaces in label files

## Quick Verification

```python
import glob

# Check training
train_imgs = len(glob.glob('data/training/*.jpg'))
train_labels = len(open('data/training/ica_lables.txt').readlines())
print(f"Training: {train_imgs} images, {train_labels} labels")

# Check test
test_imgs = len(glob.glob('data/test/*.jpg'))
test_labels = len(open('data/test/ica_lables.txt').readlines())
print(f"Test: {test_imgs} images, {test_labels} labels")
```
