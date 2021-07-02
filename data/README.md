# Data Path 

/scratch/visual/ashestak/oai/v00/data/inputs/

# Annotations Path 

/scratch/visual/ashestak/oai/v00/data/moaks 

on z1 replace 'visual' by 'htc'

# Annotations

- train.json 
- val.json 
- test.json 

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
