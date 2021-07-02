# Segmentation Masks:

root directory  
/vis/scratchN/bzftacka/OAI_DESS_Data_AllTPs/Merged/v00/OAI/

## segmentation labels (menisci)

| Side   | Label  | Meniscus |
| ------ | ------ | -------- |
| left   |    5   | medial   |
|        |    6   | lateral  |
| right  |    5   | lateral  |
|        |    6   | medial   |

=> when generating boxes you get: 

for the left side:  
> boxes = [medial, lateral]

for the right side: 
> boxes = [lateral, medial]

=> to get correct targets from annotations

```python
img = np.load(...) # load the image
tgt = {
    "labels": [anns.get("LAT"), anns.get("MED")], # for binary
    # OR for multilabel
    "labels": [
        [
            anns.get("V00MMTLA"),
            anns.get("V00MMTLB"),
            anns.get("V00MMTLP"),
        ],
        [
            anns.get("V00MMTMA"),
            anns.get("V00MMTMB"),
            anns.get("V00MMTMP"),
        ]
    ]
    "boxes": anns.get("boxes")
}

if anns.get("side") == "left":
    img = img.flip(0)
    tgt["boxes"] = tgt["boxes"][::-1]
```



# Data Path 

/scratch/visual/ashestak/oai/v00/data/inputs/

# Annotations Path 

/scratch/visual/ashestak/oai/v00/data/moaks 

on z1 replace 'visual' by 'htc'

# MOAKS 

File moaks_complete.csv

can use that to sample and generate your own splits (annotations)

# Annotations

Anotations are json files with image id as keys (see example below)  

### splits: 

- train.json 
- val.json 
- test.json

```json
annotations = {
    // ...
"10934012": {
    "dicom_dir": "/vis/scratchN/oaiDataBase/v00/OAI/0.E.1/9028904/20050621/10934012/",
    "patient_id": 9028904,
    "image_id": 10934012,
    "side": "right",
    "boxes": [
        [35, 221, 138, 84, 255, 271],   // boxes in xyxy format 
        [97, 203, 162, 143, 247, 265]   // see torchvision.ops for box formats (also below)
    ], 
    "labels": [0, 1],
    "ID": 9028904,
    "SIDE": "right",
    "READPRJ": "65", // Moaks scores for menisci 
    "V00MMTMA": 0.0, // Medial  Anterior Horn
    "V00MMTMB": 1.0, // Medial Body
    "V00MMTMP": 3.0, // Medial Posterior Horn
    "V00MMTLA": 0.0, // Lateral Anterior Horn
    "V00MMTLB": 0.0, // Lateral Body
    "V00MMTLP": 1.0, // Lateral Posterior Horn
    "PATH": "0.E.1/9028904/20050621/10934012/", // relative path to dicom
    "MED": 1,  // overall score for medial meniscus
    "LAT": 0  // overall score for lateral meniscus
    }
    // ...
}
```

# Datasets: 

### MoaksDatasetBinaryMonolabel 
generates targets (dict) that contain 
- 3d boxes in xyxy format 
- binary label for the whole menisci (medial and lateral) i.e one label for each meniscus 

Example:   
```
target = {  
    "labels": [0, 1]  # medial, lateral
    "boxes": [  
        [zmin1, ymin1, xmin1, zmax1, ymax1, xmax1],  # medial
        [zmin2, ymin2, xmin2, zmax2, ymax2, xmax2]   # lateral
    ]  
}
```

### MoaksDatasetBinaryMultilabel

 - binary multilabel for each meniscus 


Example: 
```
target = {
    "boxes":  ... ,
    "labels: [
        [0, 1, 1], # Anterior Horn, Body, Posterior Horn (medial)
        [0, 0, 1]  # lateral
    ]

}
```

# To use with dataloaders: 

```python 
data_dir = "/scratch/htc/ashestak/oai/v00/data/inputs"
annotations ="/scratch/htc/ashestak/oai/v00/data/moaks/train.json"

dataset = MOAKSDatasetBinaryMultilabel(data_dir, annotations)

dataloader = torch.utils.data.DataLoader(dataset, **kwargs)

```
