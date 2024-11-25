
for type in random popular adversarial
    do 
        /workspace/conda_env/osprey/bin/python pope_eval.py --model ./checkpoints/osprey-model \
        --img ./data/coco_imgs --json ./playground/data/eval/pope/coco/coco_pope_${type}.json \
        --answer pope/coco_pope_${type}_answers.json 
done

for type in random popular adversarial
    do 
        echo "Evaluating pope on ${type} data..."
        /workspace/conda_env/osprey/bin/python pope/evaluate.py --ans-file pope/coco_pope_${type}_answers.json  \
        --label-file ./playground/data/eval/pope/coco/coco_pope_${type}.json
done