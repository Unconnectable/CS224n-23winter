# ass1作业的所有的提示和要求

## readme的翻译

欢迎来到CS224N！

本课程将全程使用Python.如果你已经配置好了Python环境,那很好！但请确保你的Python版本至少是3.5.如果尚未安装,最简单的办法是确保你的电脑至少有3GB可用空间,然后访问(https://www.anaconda.com/download/)并安装Python 3版本的Anaconda.该安装包适用于所有操作系统.

安装完conda后,请关闭所有已打开的终端窗口.然后新建一个终端并运行以下命令：

### 1. 使用env.yml中的依赖项创建环境：

```sh
conda env create -f env.yml
```

### 2. 激活新创建的环境：

    conda activate cs224n

### 3. 在新环境中安装IPython内核,以便在jupyter notebook中使用该环境： 

    python -m ipykernel install --user --name cs224n


### 4. 作业1(仅限第一次作业)是Jupyter Notebook.完成上述步骤后,你可以通过以下命令开始：

    jupyter notebook exploring_word_vectors.ipynb

### 5. 为确保我们使用正确的环境,在exploring_word_vectors.ipynb的工具栏中点击Kernel -> Change kernel,你应该能在下拉菜单中看到并选择cs224n.

### 要停用当前激活的环境,请使用：

    conda deactivate