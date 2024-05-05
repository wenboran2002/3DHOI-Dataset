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

notice that we only keep objects with direct contact between people. For BEHAVE, we set the threshold as 30 points.


### Human:
We use SMPL-X parameters to represent each human.
if the custom dataset provide other models, you should convert it to SMPL-X format.
[convert from this repo.](https://github.com/wenboran2002/smplx)

#### contact:
We use body part label for contact regions.
We provide an example conversion for BEHAVE here.
### Interaction:

If the custom dataset does not provide interaction label, you need to generate interaction using rgba image.

Assuming that the rgba image only includes one pair of human and object, we use [Llava](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file) to generate verbs.


