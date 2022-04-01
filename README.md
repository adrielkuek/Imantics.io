![Imantics.io](https://github.com/adrielkuek/Imantics.io/blob/main/figures/Imantics_io_logo.jpg)

# Overview
Imantics.io is the idea of building a web-based image archival and search framework to simulate an intelligent mobile phone image gallery.</br>The core concept behind Imantics.io is to provide users with an
intelligent image pattern discovery framework to derive semantic understanding for archival and organisation. The framework is designed as a discovery pipeline beginning with an image dataset input, followed by image feature extraction and feature selection blocks to obtain signal representations for downstream tasks of image clustering and image retrieval respectively. The corresponding image outputs enable the system to discover novel semantics or genres of images through clustering, as well as performing zero-shot classification through retrieval. The discovery pipeline for Imantics.io is illustrated below.

![Pipeline](https://github.com/adrielkuek/Imantics.io/blob/main/figures/imantics_io_pipeline.png)

# Running Model
Go to __Command Prompt__.

```shell
cd Project                  # Change directory to Project Folder Directory
virtual\Scripts\activate    # Activate the virtual environment
python program.py           # Run the program

Go to Chrome Browser and key in the URL: http://localhost:5000/
Lastly, enjoy the tour!
```

### DataSet

Consists of 2,000 images curated from actual users' mobile phone gallery. <br />
Note: dataset is not uploaded due to massive size

### Missing Documents (File size too big to be uploaded)

1) models folder <br />
    => Contains dino_deitsmall8_pretrain.pth & ViT-B-32.pt 
2) static/images folder <br />
    => Contains 2,000 images curated from mobile phone gallery
3) virtual folder <br />
    => Virtual Environment with libraries and packages not uploaded
