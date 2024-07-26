from random import random
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from functools import partial
from llava.mm_utils import tokenizer_image_token

def call_llava_engine_df(args, sample, model, tokenizer=None, processor=None, h_block=None, w_block=None, mode=None):

    def deal_with_prompt(input_text, mm_use_im_start_end, ocr_tokens):
        if ocr_tokens is not None:
            qs = input_text + '\n' + ocr_tokens
        else:
            qs = input_text
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

    prompt = sample['final_input_prompt']
    ocr_tokens = sample.get('ocr', None)
    prompt = deal_with_prompt(prompt, model.config.mm_use_im_start_end, ocr_tokens)
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image = sample['image']

    if image is not None:
        model.orig_forward = model.forward
        model.forward = partial(model.orig_forward,
                            mode=mode,
                            h_block = [h_block],
                            w_block = [w_block]
                            )
        output_ids = model.generate(
            input_ids,
            images=image.bfloat16().cuda(),
            do_sample=False,
            temperature=0,
            num_beams=1,
            top_p=None,
            max_new_tokens=1024,
            use_cache=True)

        model.forward = model.orig_forward

        input_token_len = input_ids.shape[1]

        response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        # response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip('\n')
    else:  # multiple images actually
        if sample['question_type'] == 'multiple-choice':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

    return response
