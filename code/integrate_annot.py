import json
import os
import shutil
import cv2
import numpy as np
img_dir="/Disk1/boran/BEHAVE_dataset/behave_human_contact/"
seq_dir="/Disk1/boran/BEHAVE_dataset/sequences/"
custom_dir="/Disk1/boran/3dhoi_BEHAVE/"
smplx_dir='/Disk1/boran/BEHAVE_dataset/behave_smplx_subset/'
obj_dir='/Disk1/boran/BEHAVE_dataset/hoi_objects/'
contact_h="/Disk1/boran/BEHAVE_dataset/behave_human_contact/"
verbs_annot=json.load(open("/Disk1/boran/LLaVA/behave_verbs_all.json"))
contact_o=json.load(open("/Disk1/boran/BEHAVE_dataset/contact_labels.json"))
obj_list=['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','street sign','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','hat','backpack','umbrella','shoe','eye glasses','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','plate','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','mirror','dining table','window','desk','toilet','door','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','blender','book','clock','vase','scissors','teddy bear','hair drier','toothbrush','hair brush']
BEHAVE_OBJECTS={
'yogamat':'yogamat',
'yogaball':'sports ball',
'trashbin':'trashbin',
'toolbox':'box',
'tablesquare':'table',
'tablesmall':'table',
'suitcase':'suitcase',
'stool':'stool',
'plasticcontainer':'box',
'monitor':'monitor',
'keyboard':'keyboard',
'chairwood':'chair',
'chairblack':'chair',
'boxtiny':'box',
'boxsmall':'box',
'boxmedium':'box',
'boxlarge':'box',
'boxlong':'box',
'basketball':'sports ball',
'backpack':'backpack',
}
def mask2box(mask_path):
    mask_png = cv2.imread(mask_path)
    mask_png = mask_png[:, :, 0]
    non_zero_indices = np.nonzero(mask_png)
    if len(non_zero_indices[0]) == 0:
        return None
    # Calculate the bounding box
    min_row = np.min(non_zero_indices[1])
    min_col = np.min(non_zero_indices[0])
    width = np.max(non_zero_indices[1]) - min_row
    height = np.max(non_zero_indices[0]) - min_col

    return np.asarray([[min_row, min_col, width, height]])


# because that BEHAVE is one person - one object dataset, if the other custom
# dataset is one person - multi objects, you shoould change the code below
def extract_annotations(file_name,h_mask,o_mask,verbs,object):
    annotation={}
    annotation["file_name"]=file_name
    h_box=mask2box(h_mask).tolist()
    o_box=mask2box(o_mask)
    if o_box is None:
        return None
    o_box=o_box.tolist()

    annotation['hoi_annotation']=[]
    annotation["annotations"]=[]
    annotation['annotations'].append({'bbox':h_box,'category':'person'})
    annotation['annotations'].append({'bbox':o_box,'category_id':BEHAVE_OBJECTS[object]})
    for verb in verbs:
        annotation['hoi_annotation'].append({'subject_id':0,'object_id':1,'action':verb})
    return annotation
def generate_dataset(img_name,suffix,tt_name):
    print(img_name)
    # return
    img_path=seq_dir+suffix+'/'+tt_name+'/k1.color.jpg'
    # if not os.path.exists(custom_dir+img_name):
    #     os.mkdir(custom_dir+img_name)
    # shutil.copy(img_path,custom_dir+img_name+'/image.jpg')
    # smplx_path=smplx_dir+img_name+".000.json"
    # shutil.copy(smplx_path,custom_dir+img_name+'/smplx_parameters.json')
    # obj_path=obj_dir+img_name+".000.ply"
    # shutil.copy(obj_path,custom_dir+img_name+'/obj_pcd.ply')
    # os.mkdir(custom_dir+img_name+'/contact_label/')
    # shutil.copy(contact_h+img_name+".json",custom_dir+img_name+'/contact_label/human_part.json')
    # obj_contact=contact_o[suffix+'/'+tt_name]
    if os.path.exists(custom_dir+img_name+'/annotations.json'):
        return
    verbs=verbs_annot[img_name]
    obj=img_name.split('_')[2]
    # with open(custom_dir+img_name+'/contact_label/obj_contact.json','w') as f:
    #     json.dump(obj_contact,f)
    human_mask=seq_dir+suffix+'/'+tt_name+'/k1.person_mask.jpg'
    obj_mask=seq_dir+suffix+'/'+tt_name+'/k1.obj_rend_mask.jpg'
    annot=extract_annotations(img_name,human_mask,obj_mask,verbs,obj)

    if annot is None:
        shutil.rmtree(custom_dir+img_name)
        return
    with open(custom_dir+img_name+'/annotations.json','w') as f:
        json.dump(annot,f)

img_list=os.listdir(img_dir)
for img_fn in img_list:
    img_fn=img_fn.split('.')[0]
    suffix = ''
    for idxxx, im in enumerate(img_fn.split('_')):
        if idxxx != (len(img_fn.split('_')) - 1):
            suffix = suffix + im + '_'
    suffix = suffix[:-1]
    # print(suffix)
    frame = img_fn.split('_')[-1].split('.')[0] + '.000'
    generate_dataset(img_fn,suffix,frame)