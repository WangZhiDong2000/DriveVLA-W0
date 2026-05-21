import pickle

with open('/data2/data/navsim/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl', 'rb') as f:
    data = pickle.load(f)
pkl_tokens = {item['token'] for item in data}

yaml_tokens = set()
in_tokens_section = False
with open('/data1/cz/projects/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml') as f:
    for line in f:
        if line.strip().startswith('tokens:'):
            in_tokens_section = True
            continue
        if in_tokens_section:
            stripped = line.strip()
            if stripped.startswith('- '):
                tok = stripped[3:].strip("'")
                yaml_tokens.add(tok)

print("pkl tokens:", len(pkl_tokens))
print("navtrain yaml tokens:", len(yaml_tokens))
print("intersection:", len(pkl_tokens & yaml_tokens))
print("pkl not in yaml:", len(pkl_tokens - yaml_tokens))
print("yaml not in pkl:", len(yaml_tokens - pkl_tokens))
print("Sample yaml token:", list(yaml_tokens)[:3])
print("Sample pkl token:", list(pkl_tokens)[:3])
