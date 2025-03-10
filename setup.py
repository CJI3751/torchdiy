from setuptools import setup, find_packages

setup(
    name="torchdiy",   # 專案名稱
    version="1.3.0", # 版本號
    packages=find_packages(),  # 包含的程式碼目錄
    description="A hobby project just like torch for learning how to design a deep learning framework.",
    long_description=open('README.md').read(),  # 專案的詳細描述
    long_description_content_type="text/markdown",
    author="ccc",
    author_email="ccckmit@gmail.com",
    url="https://github.com/ccc-py/torchdiy",  # 專案的網址
    classifiers=[  # PyPI分類
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[  # 需要安裝的依賴包
        "numpy", "torch", "transformers" # 範例依賴包
    ],
    python_requires='>=3.6',  # 支援的 Python 版本
)
