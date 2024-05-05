import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import json
def mask_img(obj_mask,h_mask,img):
    obj_mask=np.asarray(obj_mask).astype(np.uint8)
    h_mask=np.asarray(h_mask).astype(np.uint8)
    om=np.zeros_like(obj_mask[:,:,0])
    hm=np.zeros_like(h_mask[:,:,0])
    om[obj_mask[:,:,0]==255]=1
    hm[h_mask[:,:,0]==255]=1
    mask=np.logical_or(hm,om)
    img=np.asarray(img).astype(np.uint8)
    img[~mask]=[255,255,255]
    return Image.fromarray(img)

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    img_dir="/Disk1/boran/BEHAVE_dataset/behave_human_contact/"
    seq_dir="/Disk1/boran/BEHAVE_dataset/sequences/"
    img_path_list=os.listdir(img_dir)
    interaction_behave='./behave_verbs.json'

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    print(conv_mode)
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode


    for img_p in tqdm(img_path_list):
        if not os.path.exists('./behave_verbs.json'):
            verbs_out={}
        else:
            verbs_out=json.load(open('./behave_verbs.json'))
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        suffix=''
        for idxxx,im in enumerate(img_p.split('_')):
            if idxxx !=( len(img_p.split('_'))-1):
                suffix = suffix+ im + '_'
        suffix=suffix[:-1]
        frame=img_p.split('_')[-1].split('.')[0]+'.000'
        # print(img_p)
        im_s_p=seq_dir+suffix+'/'+frame+'/k1.color.jpg'
        h_mask_p=seq_dir+suffix+'/'+frame+'/k1.person_mask.jpg'
        o_mask_p=seq_dir+suffix+'/'+frame+'/k1.obj_rend_mask.jpg'
        obj_name=img_p.split('_')[2]
        image = load_image(im_s_p)
        image=mask_img(load_image(o_mask_p),load_image(h_mask_p),image)
        image.save('./test.png')
        image_size = image.size
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # while True:
        #     try:
        #         inp = input(f"{roles[0]}: ")
        #     except EOFError:
        #         inp = ""
        #     if not inp:
        #         print("exit...")
        #         break
        #     print(f"{roles[1]}: ", end="")
        inp=f"{roles[0]}:choose verbs from 'chase,dry,hold,pet,board,catch,drive,assemble,straddle,jump,sit at,zip,adjust,clean,lie on,drink with,check,feed,blow,ride,stand on,wear,break,brush with,direct,run,hug,cook,sit on,cut,cut with,drag,hop on,carry,dribble,buy,fly,control,fill,repair,read,type on,row,eat,wield,operate,kick,turn,stand under,hit,throw,text on,sip,walk,eat at,point,pull,grind,block,open,greet,teach,race,stick,lick,load,toast,shear,make,stop at,flip,peel,park,push,lasso,exit,launch,wash,slide,lift,pick_up,groom,train,hose,swing,scratch,herd,sail,hunt,pour,milk,light,pick,kiss,tie,paint,install,serve,set,stir,smell,pack,release,pay,move,stab,spin,squeeze'to describe the interaction between the person and the {obj_name}.Answer like 'run,sleep' if the person is running and sleep with the object"

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)
        # exit(0)
        outputs = tokenizer.decode(output_ids[0]).strip()
        verbs_out[img_p.split('.')[0]]=outputs
        with open('./behave_verbs.json', 'w') as f:
            json.dump(verbs_out, f)
        # conv.messages[-1][-1] = outputs

        # if args.debug:
        #     print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/hdd/boran/gio/Video-LLaVA/weights/llava/llava-v1.5-7b/")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true",default=True)
    args = parser.parse_args()
    main(args)
