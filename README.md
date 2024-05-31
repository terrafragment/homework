```
# 暴力图像分类器

本项目包含一个用于使用预训练的暴力分类器模型对图像进行分类的脚本。该模型基于 PyTorch，并使用 `model.py` 中定义的自定义 `ViolenceClassifier` 模型。

## 环境要求

- Python 3.6+
- PyTorch
- torchvision
- Pillow


```

## 使用方法

要对图像进行分类，你需要一个预训练的模型检查点文件以及一个或多个要分类的图像。

### 命令行使用

该脚本从命令行运行，需要提供模型检查点的路径和要分类的图像路径。

```bash
python classify.py <checkpoint_path> <image_path1> <image_path2> ...
```

- `<checkpoint_path>`：预训练模型检查点文件的路径。
- `<image_path1> <image_path2> ...`：要分类的图像路径。

### 示例

```bash
python classify.py checkpoint.pth image1.jpg image2.jpg
```

### 输出

脚本将输出一个预测列表，每个预测对应于输入图像的类别，其中0代表非暴力图像，1代表含有暴力内容的图像 。

