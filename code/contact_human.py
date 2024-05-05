import open3d as o3d
import numpy as np
import json
import os
import torch
import smplx
import igl
from tqdm import tqdm

smplx_dir="/Disk1/boran/BEHAVE_dataset/behave_smplx_subset/"
obj_dir='/Disk1/boran/BEHAVE_dataset/hoi_objects/'
smplx_param_list=os.listdir(smplx_dir)

#### create smplx model ####
model_type='smplx'
model_folder="/hdd/boran/models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.npz"
layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
model = smplx.create(model_folder, model_type=model_type,
                         gender='neutral',
                         num_betas=10,
                         num_expression_coeffs=10,use_pca=False,use_face_contour=True,**layer_arg)
zero_pose = torch.zeros((1, 3)).float().repeat(1, 1)
############################

#### load contact part label ####
contact_output_dir="/Disk1/boran/BEHAVE_dataset/behave_human_contact/"
object_contact_anno=json.load(open("/Disk1/boran/BEHAVE_dataset/contact_labels.json"))
#################################

device='cuda'
def get_contact_labels(smplv,smplf, obj, thres=0.02):
    #####
    ## input smplx vertices,faces and object vertices, all in ndarray
    dist, _, vertices = igl.signed_distance(obj, smplv, smplf, return_normals=False)
    contact_result = dist < thres
    verts=torch.from_numpy(vertices[contact_result]).to(device)
    smplv=torch.from_numpy(smplv).to(device)
    distances = torch.sum((smplv[:, np.newaxis, :] - verts) ** 2, dim=-1)
    # Find the index of the minimum distance for each element in `elements`
    indices = torch.argmin(distances, dim=0)
    ## return object_contact_indices, corresponding_human_verts, num_object_contact_points,corresponding_human_verts_index
    return dist < thres, vertices, np.sum(contact_result),indices.cpu().detach().numpy()

template_parts=json.load(open("/hdd/boran/HOIcontact/annotation/final_part_id_pcd.json"))

total=0
for smplx_param_name in tqdm(smplx_param_list):
    key_n=smplx_param_name.split('.')[0]
    smplx_path=os.path.join(smplx_dir,smplx_param_name)
    smpl_param=json.load(open(smplx_path))
    print(key_n)

    #### slightly change smplx parameters(optional for other dataset)
    for k,v in smpl_param.items():
        if k=='shape':
            v=v[:10]
        v=np.asarray(v).reshape(1,-1).astype(np.float32)
        smpl_param[k]=v

    output = model(betas=torch.tensor(smpl_param['shape']), body_pose=torch.tensor(smpl_param['body_pose']),
                   global_orient=torch.tensor(smpl_param['root_pose']),
                   right_hand_pose=torch.tensor(smpl_param['rhand_pose']),
                   left_hand_pose=torch.tensor(smpl_param['lhand_pose']), jaw_pose=torch.tensor(smpl_param['jaw_pose']),
                   leye_pose=zero_pose,
                   reye_pose=zero_pose, expression=torch.tensor(smpl_param['exp']))
    object=o3d.io.read_point_cloud(obj_dir+key_n+'.000.ply')
    object_points=np.asarray(object.points)
    transl=smpl_param['cam_trans']
    human_points=np.asarray(output.vertices).squeeze(0)+transl
    faces=np.asarray(model.faces).astype(np.int32)
    contact_o_idx,vertices,num_contact,hidx=get_contact_labels(human_points,faces,object_points)
    # print(hidx.shape)

    #### filter
    if num_contact<50:
        continue
    part_list=[]
    for iidxx in hidx:
        if template_parts[iidxx] not in part_list:
            part_list.append(template_parts[iidxx])
    output_path=contact_output_dir+key_n+'.json'
    with open(output_path,'w') as f:
        json.dump(part_list,f)
        total+=1

print('total',total)