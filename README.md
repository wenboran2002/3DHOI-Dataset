# 3DHOI-Dataset
transform custom dataset to 3d hoi dataset format

## 3DHOI Dataset Format
this is the implementation of converting custom 3d hoi dataset to our format, 
we use BEHAVE as the example.
### Object:
each object contains 4096 points,
for custom dataset, extract objects in the frame and randomly select 4096 points.

#### contact:
binary contact label

notice that we only keep objects with direct contact between people. For BEHAVE, we set the threshold as 50 points.

```python
import igl
import numpy as np
def get_contact_labels(self, smpl, obj, num_samples, thres=0.02):
    """
    sample point on the object surface and compute contact labels for each point
    :param smpl: trimesh object
    :param obj: trimesh object
    :param num_samples: number of samples on object surface
    :param thres: threshold to determine whether a point is in contact with the human
    :return:
    for each point: a binary label (contact or not) and the closest SMPL vertex
    """
    object_points = obj.sample(num_samples)
    dist, _, vertices = igl.signed_distance(object_points, smpl.vertices, smpl.faces, return_normals=False)
    contact_result = dist < thres
    # print('num of contact points:', np.sum(contact_result))
    return object_points, dist < thres, vertices, np.sum(contact_result)
```

### Human:
We use SMPL-X parameters to represent each human.
if the custom dataset provide other models, you should convert it to SMPL-X format.
[convert from this repo.](https://github.com/wenboran2002/smplx)

We need SMPL-X parameters below(the final keys should be the same as keys below):
```text
'body_pose',
'lhand_pose',
'shape',
'root_pose',
'rhand_pose',
'jaw_pose', 
'exp', 
'cam_trans'
```
#### contact:
We use body part label for contact regions.
We provide an example conversion for BEHAVE here.
### Interaction:

If the custom dataset does not provide interaction label, you need to generate interaction using rgba image.

Assuming that the rgba image only includes one pair of human and object, we use [Llava](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file) to generate verbs.


